import math


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with optional warmup.

    The *_restart schedules are meant for fine-tuning from a checkpoint: warmup
    and decay are measured from args.start_epoch instead of epoch 0.
    """
    restart_epoch = getattr(args, "lr_restart_epoch", -1)
    if restart_epoch < 0:
        restart_epoch = getattr(args, "start_epoch", 0)

    is_restart = args.lr_schedule.endswith("_restart")
    schedule = args.lr_schedule[:-8] if is_restart else args.lr_schedule
    schedule_epoch = epoch - restart_epoch if is_restart else epoch
    schedule_epochs = args.epochs - restart_epoch if is_restart else args.epochs
    schedule_epoch = max(0.0, schedule_epoch)
    schedule_epochs = max(args.warmup_epochs + 1e-8, schedule_epochs)

    if args.warmup_epochs > 0 and schedule_epoch < args.warmup_epochs:
        lr = args.lr * schedule_epoch / args.warmup_epochs
    else:
        if schedule == "constant":
            lr = args.lr
        elif schedule == "cosine":
            progress = (schedule_epoch - args.warmup_epochs) / (schedule_epochs - args.warmup_epochs)
            progress = min(1.0, max(0.0, progress))
            lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
                (1. + math.cos(math.pi * progress))
        elif schedule == "flat_linear_cosine":
            linear_start = getattr(args, "lr_linear_start_epoch", 160.0)
            linear_end = getattr(args, "lr_linear_end_epoch", 162.0)
            linear_end_lr = getattr(args, "lr_linear_end_lr", 2e-5)

            if schedule_epoch < linear_start:
                lr = args.lr
            elif schedule_epoch < linear_end:
                progress = (schedule_epoch - linear_start) / max(1e-8, linear_end - linear_start)
                progress = min(1.0, max(0.0, progress))
                lr = args.lr + (linear_end_lr - args.lr) * progress
            else:
                progress = (schedule_epoch - linear_end) / max(1e-8, schedule_epochs - linear_end)
                progress = min(1.0, max(0.0, progress))
                lr = args.min_lr + (linear_end_lr - args.min_lr) * 0.5 * \
                    (1. + math.cos(math.pi * progress))
        else:
            raise NotImplementedError
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
