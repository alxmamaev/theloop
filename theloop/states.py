class State:
    def __init__(self, model=None, device=None, summary_writer=None,
                callback_func=None):
        self.model = model
        self.device = device
        self.it = 0
        self.metrics = {}
        self.writer = summary_writer
        self.callback_func = callback_func

    @staticmethod
    def tb_log(writer, data, it):
        for k, v in data.items():
            writer.add_scalar(k, float(v), it)

    def set_metric(self, name, value):
        self.metrics[name] = value

    def write_metrics(self):
        self.tb_log(self.writer, self.metrics, self.it)
        self.it += 1


class BatchState(State):
    def __init__(self,
                model=None,
                device=None,
                summary_writer=None,
                criterion=None,
                optimizer=None,
                callback_func=None):

        super().__init__(model=model,
                         device=device,
                         summary_writer=summary_writer,
                         callback_func=callback_func)

        self.optimizer = optimizer
        self.criterion = criterion
        self.loss = None
        self.batch = None


    def step(self, batch):
        self.batch = batch
        self.metrics = {}

        self.callback_func(self)
        self.set_metric("loss", self.loss.item())
        self.write_metrics()

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()


class ValidationState(State):
    def __init__(self, model=None,
                device=None, summary_writer=None,
                callback_func=None,
                val_criterion_mode="max",
                val_criterion_key=None):

        super().__init__(model=model,
                         device=device,
                         summary_writer=summary_writer,
                         callback_func=callback_func)
        self.data = None
        self._best_checkpoint = None
        self._best_checkpoint_score = None
        self._best_checkpoint_metrics = None
        self.val_criterion_mode = val_criterion_mode
        self.val_criterion_key = val_criterion_key



    def step(self, data):
        self.data = data
        self.metrics = {}
        self.callback_func(self)

        if self.val_criterion_key is not None:
            score = float(self.metrics[self.val_criterion_key])

            best_checkpoint_exist = self.best_checkpoint is not None
            val_criterion_mode = self.val_criterion_mode == "min"
            score_less_best_score =  score < self.best_checkpoint_score

            if not best_checkpoint_exist or (val_criterion_mode == score_less_best_score):
                self._best_checkpoint = self.model.state_dict()
                self._best_checkpoint_score = score
                self._best_checkpoint_metrics = self.metrics

        self.write_metrics()
