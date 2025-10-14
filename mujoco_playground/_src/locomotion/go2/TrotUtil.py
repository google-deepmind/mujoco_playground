import jax.numpy as jp

# ----------------- utils -----------------
def cos_wave(t, step_period, scale):
    _cos_wave = -jp.cos(((2 * jp.pi) / step_period) * t)
    return _cos_wave * (scale / 2) + (scale / 2)


def dcos_wave(t, step_period, scale):
    return ((scale * jp.pi) / step_period) * jp.sin(((2 * jp.pi) / step_period) * t)


def make_kinematic_ref(sinusoid, step_k, scale=0.3, dt=1/50):
    _steps = jp.arange(step_k)
    step_period = step_k * dt
    t = _steps * dt

    wave = sinusoid(t, step_period, scale)
    fleg_cmd_block = jp.concatenate(
        [jp.zeros((step_k, 1)),
         wave.reshape(step_k, 1),
         -2 * wave.reshape(step_k, 1)],
        axis=1
    )
    h_leg_cmd_bloc = +1 * fleg_cmd_block # identical legs (go2): +1;   mirrored legs (anymal): -1

    block1 = jp.concatenate([
        jp.zeros((step_k, 3)),
        fleg_cmd_block,
        h_leg_cmd_bloc,
        jp.zeros((step_k, 3))],
        axis=1
    )

    block2 = jp.concatenate([
        fleg_cmd_block,
        jp.zeros((step_k, 3)),
        jp.zeros((step_k, 3)),
        h_leg_cmd_bloc],
        axis=1
    )

    step_cycle = jp.concatenate([block1, block2], axis=0)
    return step_cycle


def quaternion_to_matrix(quaternions):
    r, i, j, k = quaternions[..., 0], quaternions[..., 1], quaternions[..., 2], quaternions[..., 3]
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = jp.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def matrix_to_rotation_6d(matrix):
    batch_dim = matrix.shape[:-2]
    return matrix[..., :2, :].reshape(batch_dim + (6,))


def quaternion_to_rotation_6d(quaternion):
    return matrix_to_rotation_6d(quaternion_to_matrix(quaternion))

def rotate(v: jp.ndarray, q: jp.ndarray) -> jp.ndarray:
    """旋转向量 v by 四元数 q（世界->旋转后的方向）"""
    # 等价于 R @ v
    R = quaternion_to_matrix(q)
    return R @ v

def rotate_inv(v: jp.ndarray, q: jp.ndarray) -> jp.ndarray:
    """把世界系向量 v 旋到局部系（等价于 R^T @ v）"""
    R = quaternion_to_matrix(q)
    return R.T @ v
# ----------------- utils end -----------------