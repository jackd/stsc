import keras


class PerFrameAccuracy(keras.metrics.Metric):
    def __init__(self, num_frames: int, name="per_frame_accuracy", **kwargs):
        self.num_frames = num_frames
        super().__init__(name=name, **kwargs)
        self.total = self.add_variable(
            shape=(num_frames,), initializer="zeros", name="total"
        )
        self.correct = self.add_variable(
            shape=(num_frames,), initializer="zeros", name="correct"
        )

    def get_config(self):
        config = super().get_config()
        config.update(num_frames=self.num_frames)
        return config

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true: [batch_size] or [batch_size, num_frames]
        # y_pred: [batch_size, num_frames, num_classes]
        # sample_weight: [batch_size] or [batch_size, num_frames]
        if len(y_true.shape) == 1:
            y_true = keras.ops.expand_dims(y_true, axis=1)
        correct = keras.ops.equal(
            keras.ops.argmax(y_pred, axis=-1), y_true
        )  # [batch_size, num_frames]
        correct = keras.ops.cast(correct, self.dtype)
        if sample_weight is None:
            self.total.assign_add(keras.ops.shape(y_true)[0])
            self.correct.assign_add(keras.ops.sum(correct, axis=0))
        else:
            if len(sample_weight.shape) == 1:
                sample_weight = keras.ops.expand_dims(sample_weight, axis=1)
            self.total.assign_add(keras.ops.sum(sample_weight, axis=0))
            self.correct.assign_add(keras.ops.sum(correct * sample_weight, axis=0))

    def result(self):
        # print(self.correct / self.total)
        return self.correct / self.total
