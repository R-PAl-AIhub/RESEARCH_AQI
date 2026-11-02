"""Part 4: five forecasters (GRU, LSTM, 1D CNN, CNN + LSTM, Temporal Fusion Transformer) that share one rule:
every output is computed from a d-sized "embedding" layer only (the notepad Worker B reads)."""
from tensorflow import keras

HEAD_LAYERS = ("forecast", "median", "gap_low", "gap_high")   # the only layers that read the embedding
KNOWN_FUTURE = ("hour_sin", "hour_cos", "doy_sin", "doy_cos")   # TFT inputs known in advance for every horizon


def recurrent_encoder(x, layer, units, dropout):
    """Stacked GRU or LSTM layers; the last one returns only its final state.

    What: returns a (batch, units[-1]) tensor that summarises the 48 hours.
    How:  every layer but the last returns the whole sequence so the next layer
          can read it; each switches off a random `dropout` share of its inputs
          during training only.
    """
    name = layer.__name__.lower()
    for i, n in enumerate(units):
        x = layer(n, dropout=dropout, return_sequences=i < len(units) - 1, name=f"{name}_{i + 1}")(x)
    return x


def conv_blocks(x, filters, kernel_size):
    """Causal 1D convolutions, each followed by max-pooling over 2 hours.

    What: returns the shorter, wider sequence (batch, hours / 2^n, filters[-1]).
    How:  "causal" padding means the filter at hour t sees hours t-k+1..t only;
          ReLU keeps the filters simple; pooling halves the hours each time.
    """
    for i, n in enumerate(filters):
        x = keras.layers.Conv1D(n, kernel_size, padding="causal", activation="relu", name=f"conv_{i + 1}")(x)
        x = keras.layers.MaxPooling1D(2, name=f"pool_{i + 1}")(x)
    return x


def gated_residual(x, hidden, dropout, name, context=None, out=None):
    """Gated residual network (GRN), the TFT building block.

    What: returns LayerNorm(skip + GLU(h)), h = Dense(ELU(Dense(x) + Dense(context))),
          width `out` (default hidden). The GLU gate (0..1) decides how much of h is added.
    How:  skip = x, or a Dense projection of x if its width is not `out`; the optional
          context (a static vector per sample) is added inside, the same for every hour.
    """
    out = out or hidden
    skip = x if x.shape[-1] == out else keras.layers.Dense(out, name=f"{name}_skip")(x)
    h = keras.layers.Dense(hidden, name=f"{name}_dense1")(x)
    if context is not None:
        c = keras.layers.Dense(hidden, use_bias=False, name=f"{name}_context")(context)
        if len(h.shape) == 3:
            c = keras.layers.RepeatVector(h.shape[1], name=f"{name}_context_hours")(c)
        h = keras.layers.Add(name=f"{name}_with_context")([h, c])
    h = keras.layers.Activation("elu", name=f"{name}_elu")(h)
    h = keras.layers.Dropout(dropout, name=f"{name}_drop")(keras.layers.Dense(hidden, name=f"{name}_dense2")(h))
    return glu_add_norm(skip, h, out, name)


def glu_add_norm(skip, new, width, name):
    """The TFT "gate, add and norm" step: LayerNorm(skip + sigmoid(Dense(new)) x Dense(new)).

    What: returns a tensor shaped like `skip`.
    How:  the gated linear unit (GLU) can switch the new layer off (gate near 0),
          so the network uses a part only when it helps.
    """
    gate = keras.layers.Dense(width, activation="sigmoid", name=f"{name}_gate")(new)
    value = keras.layers.Dense(width, name=f"{name}_value")(new)
    gated = keras.layers.Multiply(name=f"{name}_gated")([gate, value])
    return keras.layers.LayerNormalization(name=f"{name}_norm")(keras.layers.Add(name=f"{name}_add")([skip, gated]))


@keras.utils.register_keras_serializable(package="aqi")
class WeightedSum(keras.layers.Layer):
    """Sum of the column embeddings weighted by the variable-selection weights.

    What: inputs [embeddings (batch, steps, columns, hidden), weights (batch,
          steps, columns, 1)] -> (batch, steps, hidden).
    How:  multiply, then sum over the columns axis. No weights of its own.
    """

    def call(self, inputs):
        embeddings, weights = inputs
        return keras.ops.sum(embeddings * weights, axis=2)


@keras.utils.register_keras_serializable(package="aqi")
class VariableGRN(keras.layers.Layer):
    """One GRN per input column, each with its own weights, applied at every step.

    What: (batch, steps, columns, hidden) -> same shape.
    How:  every weight has a column axis, so einsum applies column j's weights to
          column j only: h = Dense2(ELU(Dense1(x))); out = LayerNorm(x + sigmoid(gate(h)) x value(h)).
    """

    def __init__(self, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.rate = dropout
        self.drop = keras.layers.Dropout(dropout)
        self.norm = keras.layers.LayerNormalization()

    def build(self, shape):
        v, h = shape[-2], shape[-1]
        init = keras.initializers.RandomUniform(-(6 / (2 * h)) ** 0.5, (6 / (2 * h)) ** 0.5)
        for name in ("w1", "w2", "wg", "wv"):
            setattr(self, name, self.add_weight(name=name, shape=(v, h, h), initializer=init))
        for name in ("b1", "b2", "bg", "bv"):
            setattr(self, name, self.add_weight(name=name, shape=(v, h), initializer="zeros"))
        self.norm.build(shape)

    def call(self, x, training=None):
        dense = lambda a, w, b: keras.ops.einsum("btvh,vhk->btvk", a, w) + b
        h = keras.activations.elu(dense(x, self.w1, self.b1))
        h = self.drop(dense(h, self.w2, self.b2), training=training)
        gated = keras.activations.sigmoid(dense(h, self.wg, self.bg)) * dense(h, self.wv, self.bv)
        return self.norm(x + gated)

    def get_config(self):
        return {**super().get_config(), "dropout": self.rate}


@keras.utils.register_keras_serializable(package="aqi")
class InterpretableAttention(keras.layers.Layer):
    """Interpretable multi-head self-attention of the TFT, causal over the steps.

    What: (batch, steps, hidden) -> (batch, steps, hidden). Each head has its own
          query and key; all heads share ONE value projection and are averaged, so
          the mean attention weights show which hours mattered.
    How:  step t only sees steps <= t (lower-triangular mask); per head
          softmax(Q K^T / sqrt(d)) V; mean over heads; Dense back to hidden.
    """

    def __init__(self, heads, hidden, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.heads, self.hidden, self.rate = heads, hidden, dropout
        self.dk = hidden // heads
        self.queries = [keras.layers.Dense(self.dk, name=f"query_{i}") for i in range(heads)]
        self.keys = [keras.layers.Dense(self.dk, name=f"key_{i}") for i in range(heads)]
        self.value = keras.layers.Dense(self.dk, name="value_shared")
        self.merge = keras.layers.Dense(hidden, name="merge")
        self.drop = keras.layers.Dropout(dropout)

    def build(self, shape):
        for layer in self.queries + self.keys + [self.value]:
            layer.build(shape)
        self.merge.build((*shape[:-1], self.dk))

    def call(self, x, training=None):
        steps = keras.ops.shape(x)[1]
        mask = keras.ops.tril(keras.ops.ones((steps, steps)))
        v = self.value(x)
        outs = []
        for q, k in zip(self.queries, self.keys):
            score = keras.ops.matmul(q(x), keras.ops.transpose(k(x), (0, 2, 1))) / self.dk ** 0.5
            score = keras.ops.where(mask > 0, score, -1e9)
            weights = self.drop(keras.ops.softmax(score, axis=-1), training=training)
            outs.append(keras.ops.matmul(weights, v))
        return self.merge(keras.ops.mean(keras.ops.stack(outs, axis=0), axis=0))

    def get_config(self):
        return {**super().get_config(), "heads": self.heads, "hidden": self.hidden, "dropout": self.rate}


def select_variables(x, hidden, dropout, context, name):
    """TFT variable selection network for time-varying inputs.

    What: (batch, steps, columns) -> (batch, steps, hidden): a weighted sum of the
          columns, with weights the model learns per sample and step.
    How:  each column gets its own linear embedding (EinsumDense, one weight row
          per column) and its own GRN (VariableGRN); a GRN on all embeddings plus
          the static selection context gives one softmax weight per column.
    """
    steps, cols = x.shape[1], x.shape[2]
    emb = keras.layers.EinsumDense("btv,vh->btvh", output_shape=(steps, cols, hidden), bias_axes="vh",
                                   name=f"{name}_embed")(x)
    processed = VariableGRN(dropout, name=f"{name}_column_grn")(emb)
    flat = keras.layers.Reshape((steps, cols * hidden), name=f"{name}_flat")(emb)
    weights = gated_residual(flat, hidden, dropout, f"{name}_select", context=context, out=cols)
    weights = keras.layers.Softmax(axis=-1, name=f"{name}_weights")(weights)
    weights = keras.layers.Reshape((steps, cols, 1), name=f"{name}_weights_4d")(weights)
    return WeightedSum(name=f"{name}_selected")([processed, weights])


def tft_encoder(x, known, station, n_stations, hidden, heads, dropout):
    """Temporal Fusion Transformer (Lim et al. 2021); its decoder output feeds the embedding.

    What: returns (batch, horizons x hidden): the decoder state of every forecast
          horizon, flattened.
    How:  (1) static covariate encoder: station -> embedding -> GRN -> four context
          vectors (selection, enrichment, LSTM start state h and c);
          (2) variable selection for the past hours (every input column) and for the
          horizons (known calendar: hour and day-of-year sin/cos at t + h);
          (3) LSTM encoder over the past (started from the static state) and LSTM
          decoder over the horizons (continues the encoder state); gate, add, norm
          with the selected inputs; (4) static enrichment GRN; (5) interpretable
          multi-head attention, causal over past + horizons; gate, add, norm;
          position-wise GRN; gate, add, norm with the LSTM output.
          Only change from the paper: the horizon outputs are not read directly, they
          go through the shared d-sized embedding (the rule Worker B relies on).
    """
    s = keras.layers.Embedding(n_stations, hidden, name="station_embed")(station)
    s = gated_residual(keras.layers.Reshape((hidden,), name="station_vector")(s), hidden, dropout, "static_grn")
    c_select, c_enrich, c_h, c_c = (gated_residual(s, hidden, dropout, f"context_{n}")
                                    for n in ("select", "enrich", "state_h", "state_c"))
    past = select_variables(x, hidden, dropout, c_select, "past")
    future = select_variables(known, hidden, dropout, c_select, "future")
    enc, h, c = keras.layers.LSTM(hidden, return_sequences=True, return_state=True, name="encoder_lstm")(
        past, initial_state=[c_h, c_c])
    dec = keras.layers.LSTM(hidden, return_sequences=True, name="decoder_lstm")(future, initial_state=[h, c])
    lstm_out = keras.layers.Concatenate(axis=1, name="lstm_steps")([enc, dec])
    selected = keras.layers.Concatenate(axis=1, name="selected_steps")([past, future])
    temporal = glu_add_norm(selected, keras.layers.Dropout(dropout, name="lstm_drop")(lstm_out), hidden, "lstm_gate")
    enriched = gated_residual(temporal, hidden, dropout, "enrich", context=c_enrich)
    att = InterpretableAttention(heads, hidden, dropout, name="attention")(enriched)
    att = glu_add_norm(enriched, keras.layers.Dropout(dropout, name="attention_drop")(att), hidden, "attention_gate")
    out = glu_add_norm(temporal, gated_residual(att, hidden, dropout, "position_grn"), hidden, "output_gate")
    horizons = keras.layers.Cropping1D((x.shape[1], 0), name="horizon_steps")(out)
    return keras.layers.Flatten(name="horizon_flat")(horizons)


def encode(name, inputs, f_cfg, dropout):
    """Run the inputs through the chosen model's body (everything before the embedding).

    What: returns a 2D tensor (batch, width) for gru, lstm, cnn or cnn_lstm (the TFT is built in build()).
    How:  layer sizes come from config.yaml forecasters.<name>; the CNN output is
          flattened (all pooled hours kept) and gets a Dropout layer, since
          Conv1D has no built-in input dropout.
    """
    cfg = f_cfg[name]
    if name == "gru":
        return recurrent_encoder(inputs, keras.layers.GRU, cfg["units"], dropout)
    if name == "lstm":
        return recurrent_encoder(inputs, keras.layers.LSTM, cfg["units"], dropout)
    if name == "cnn":
        x = keras.layers.Flatten(name="flatten")(conv_blocks(inputs, cfg["filters"], cfg["kernel_size"]))
        return keras.layers.Dropout(dropout, name="drop")(x)
    if name == "cnn_lstm":
        x = conv_blocks(inputs, cfg["filters"], cfg["kernel_size"])
        return recurrent_encoder(x, keras.layers.LSTM, cfg["lstm_units"], dropout)
    raise ValueError(f"unknown forecaster: {name}")


@keras.utils.register_keras_serializable(package="aqi")
class PinballLoss(keras.losses.Loss):
    """Pinball (quantile) loss for the [low, median, high] outputs.

    What: for quantile q, a miss above the line costs q x miss and a miss below
          costs (1 - q) x miss. With q = 0.05 the low line is pushed down until
          only ~5% of real values fall below it; q = 0.5 gives the median.
    How:  y_pred holds 3 x n_out numbers (all low, all median, all high), y_true
          is (samples, targets, horizons); both are reshaped so each line is
          compared with the same real value; averaged over lines and outputs.
    """

    def __init__(self, quantiles, n_out, name="pinball", **kwargs):
        super().__init__(name=name, **kwargs)
        self.quantiles, self.n_out = list(quantiles), n_out

    def call(self, y_true, y_pred):
        pred = keras.ops.reshape(y_pred, (-1, 3, self.n_out))
        miss = keras.ops.expand_dims(keras.ops.reshape(y_true, (-1, self.n_out)), 1) - pred
        q = keras.ops.reshape(keras.ops.convert_to_tensor(self.quantiles, dtype=pred.dtype), (1, 3, 1))
        return keras.ops.mean(keras.ops.maximum(q * miss, (q - 1) * miss), axis=(1, 2))

    def get_config(self):
        return {**super().get_config(), "quantiles": self.quantiles, "n_out": self.n_out}


def build(name, n_hours, n_features, n_targets, n_horizons, d, f_cfg, m_cfg, alpha=None, n_stations=1):
    """One forecaster plus a second model that returns its embedding.

    What: returns (model, embed_model).
          alpha None -> point model: output (batch, targets, horizons), MSE loss.
          alpha set  -> quantile model: output 3 x targets x horizons numbers in the
          order [all low, all median, all high], pinball loss for alpha/2, 0.5,
          1 - alpha/2.
    How:  body (encode; for the TFT tft_encoder with three inputs: the last hours,
          the known future calendar per horizon and the station index) ->
          Dense(d, tanh) named "embedding" -> heads. The heads
          read the embedding ONLY (no skip connection), so the forecast is a
          function of the d numbers alone, which Worker B relies on. tanh keeps
          every unit active (ReLU can leave units stuck at 0 and break the
          covariance in Part 5). low = median - softplus(.), high = median +
          softplus(.): softplus is never negative, so low <= median <= high.
          Optimiser Adam with the shared learning rate.
    """
    n_out = n_targets * n_horizons
    x_in = keras.Input(shape=(n_hours, n_features), name="last_hours")
    if name == "tft":
        inputs = [x_in, keras.Input(shape=(n_horizons, len(KNOWN_FUTURE)), name="known_future"),
                  keras.Input(shape=(1,), dtype="int32", name="station")]
        body = tft_encoder(*inputs, n_stations, f_cfg["tft"]["hidden"], f_cfg["tft"]["heads"], m_cfg["dropout"])
    else:
        inputs, body = x_in, encode(name, x_in, f_cfg, m_cfg["dropout"])
    embedding = keras.layers.Dense(d, activation="tanh", name="embedding")(body)
    optimizer = keras.optimizers.Adam(m_cfg["learning_rate"])

    if alpha is None:
        flat = keras.layers.Dense(n_out, name="forecast")(embedding)
        output = keras.layers.Reshape((n_targets, n_horizons), name="by_target")(flat)
        model = keras.Model(inputs, output, name=f"{name}_point")
        model.compile(optimizer=optimizer, loss="mse")
    else:
        median = keras.layers.Dense(n_out, name="median")(embedding)
        gap_low = keras.layers.Dense(n_out, activation="softplus", name="gap_low")(embedding)
        gap_high = keras.layers.Dense(n_out, activation="softplus", name="gap_high")(embedding)
        low = keras.layers.Subtract(name="low")([median, gap_low])
        high = keras.layers.Add(name="high")([median, gap_high])
        output = keras.layers.Concatenate(name="low_median_high")([low, median, high])
        model = keras.Model(inputs, output, name=f"{name}_quantile")
        model.compile(optimizer=optimizer, loss=PinballLoss([alpha / 2, 0.5, 1 - alpha / 2], n_out))
    return model, keras.Model(inputs, embedding, name=f"{model.name}_embedding")
