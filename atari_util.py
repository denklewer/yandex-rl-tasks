import atari_wrappers

def PrimaryAtariWrap(env,
                     clip_rewards=True,
                     frame_skip=True,
                     fire_reset_event=False,
                     width=44,
                     height=44,
                     margins=[1,1,1,1],
                     n_frames=4):

    # This wrapper holds the same action for <skip> frames and outputs
    # the maximal pixel value of 2 last frames (to handle blinking
    # in some envs)
    if frame_skip:
        env = atari_wrappers.MaxAndSkipEnv(env, skip=4)

    # This wrapper sends done=True when each life is lost
    # (not all the 5 lives that are givern by the game rules).
    # It should make easier for the agent to understand that losing is bad.
    env = atari_wrappers.EpisodicLifeEnv(env)

    # This wrapper laucnhes the ball when an episode starts.
    # Without it the agent has to learn this action, too.
    # Actually it can but learning would take longer.
    if fire_reset_event:
               env = atari_wrappers.FireResetEnv(env)

    # This wrapper transforms rewards to {-1, 0, 1} according to their sign
    if clip_rewards:
        env = atari_wrappers.ClipRewardEnv(env)

    # This wrapper is yours :)
    env = atari_wrappers.PreprocessAtariObs(env,height=height, width=width, margins=margins)

    env = atari_wrappers.FrameBuffer(env, n_frames=n_frames, dim_order='pytorch')

    return env
