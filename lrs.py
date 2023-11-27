from bisect import bisect_right
import math

class LR_Scheduler:
    """
    A learning rate scheduler that adjusts the learning rate based on the defined schedule.
    """
    
    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0, lr_step=0, warmup_epochs=0):
        """
        Initializes the learning rate scheduler.
        
        :param mode: The mode of the scheduler ('cos', 'poly', 'step').
        :param base_lr: Base learning rate.
        :param num_epochs: Total number of epochs.
        :param iters_per_epoch: Number of iterations per epoch.
        :param lr_step: Step size for 'step' mode.
        :param warmup_epochs: Number of warm-up epochs.
        """
        self.mode = mode
        self.base_lr = base_lr
        self.num_epochs = num_epochs
        self.iters_per_epoch = iters_per_epoch
        self.lr_step = lr_step
        self.warmup_epochs = warmup_epochs
        self.warmup_iters = warmup_epochs * iters_per_epoch
        self.total_iters = (num_epochs - warmup_epochs) * iters_per_epoch
        self.current_epoch = -1

    def __call__(self, optimizer, iteration, epoch, best_pred):
        """
        Adjusts the learning rate based on the current epoch and iteration.
        
        :param optimizer: The optimizer for which to adjust the learning rate.
        :param iteration: Current iteration.
        :param epoch: Current epoch.
        :param best_pred: Current best prediction (not used in this function).
        """
        current_iter = epoch * self.iters_per_epoch + iteration
        lr = self.calculate_lr(current_iter)

        # Warm-up learning rate schedule
        if self.warmup_iters > 0 and current_iter < self.warmup_iters:
            lr = lr * current_iter / self.warmup_iters

        if epoch > self.current_epoch:
            self.current_epoch = epoch

        assert lr >= 0, "Learning rate should be non-negative"
        self.adjust_learning_rate(optimizer, lr)

    def calculate_lr(self, current_iter):
        """
        Calculates learning rate based on the current iteration and the scheduling mode.
        
        :param current_iter: Current iteration.
        :return: Calculated learning rate.
        """
        if self.mode == 'cos':
            lr = 0.5 * self.base_lr * (1 + math.cos(1.0 * (current_iter - self.warmup_iters) / self.total_iters * math.pi))
        elif self.mode == 'poly':
            lr = self.base_lr * pow((1 - 1.0 * (current_iter - self.warmup_iters) / self.total_iters), 0.9)
        elif self.mode == 'step':
            lr = self.base_lr * (0.1 ** (self.current_epoch // self.lr_step))
        else:
            raise NotImplementedError(f"LR schedule mode '{self.mode}' is not implemented.")
        
        return lr

    def adjust_learning_rate(self, optimizer, lr):
        """
        Adjusts the learning rate for all parameter groups in the optimizer.
        
        :param optimizer: The optimizer for which to adjust the learning rate.
        :param lr: New learning rate.
        """
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
