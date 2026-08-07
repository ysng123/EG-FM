# Modified from https://github.com/NVlabs/Sana

from diffusion.model import gaussian_diffusion as gd
from diffusion.model.respace import SpacedDiffusion, space_timesteps
from egfm import EnergyGuidedPath


def Scheduler(
    timestep_respacing,
    noise_schedule="linear",
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    predict_flow_v=False,
    learn_sigma=True,
    pred_sigma=True,
    rescale_learned_sigmas=False,
    diffusion_steps=1000,
    snr=False,
    return_startx=False,
    flow_shift=1.0,
    egfm_sigma0=3.5,
    egfm_curve="smootherstep",
    egfm_release_start=0.0,
    egfm_release_end=1.0,
):
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    if predict_xstart:
        model_mean_type = gd.ModelMeanType.START_X
    elif predict_flow_v:
        model_mean_type = gd.ModelMeanType.FLOW_VELOCITY
    else:
        model_mean_type = gd.ModelMeanType.EPSILON
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=model_mean_type,
        model_var_type=(
            (
                (gd.ModelVarType.FIXED_LARGE if not sigma_small else gd.ModelVarType.FIXED_SMALL)
                if not learn_sigma
                else gd.ModelVarType.LEARNED_RANGE
            )
            if pred_sigma
            else None
        ),
        loss_type=loss_type,
        snr=snr,
        return_startx=return_startx,
        flow="flow" in noise_schedule,
        flow_shift=flow_shift,
        diffusion_steps=diffusion_steps,
        egfm_path=EnergyGuidedPath(
            sigma0=egfm_sigma0,
            curve=egfm_curve,
            release_start=egfm_release_start,
            release_end=egfm_release_end,
        ),
    )
