import numpy as np
import numpy.linalg as linalg
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize


joints_names = [
    "nose",
    "head",
    "neck",
    "l_shoulder",
    "l_elbow",
    "l_wrist",
    "lf_paw",
    "r_shoulder",
    "r_elbow",
    "r_wrist",
    "rf_paw",
    "l_hip",
    "l_knee",
    "l_ankle",
    "lb_paw",
    "r_hip",
    "r_knee",
    "r_ankle",
    "rb_paw",
]


def get_bones(d2_keypoints: list[np.ndarray]):
    opti_params = None
    for progress, opti_res in optimize(d2_keypoints):
        if opti_res is None:
            yield progress, None
        else:
            opti_params = opti_res
    scale, rot, _, shape_scalar_params, shape_angle_params = _unpack_params(opti_params)
    joints: list[np.ndarray] = _construct_model_joints(
        shape_scalar_params, shape_angle_params
    )
    scaled_joints = _scale_joints(joints, scale)
    joints = [np.array(j) for j in _rotate3d_joints(scaled_joints, rot)]

    nj = dict()
    for i in range(len(joints_names)):
        nj[joints_names[i]] = joints[i]

    yield 100, [
        # neck up
        (nj["nose"], nj["head"]),
        (nj["head"], nj["neck"]),
        (nj["neck"], nj["l_shoulder"]),
        (nj["neck"], nj["r_shoulder"]),
        (nj["neck"], nj["r_hip"]),
        (nj["neck"], nj["l_hip"]),
        # frame
        (nj["l_shoulder"], nj["l_hip"]),
        (nj["l_shoulder"], nj["r_shoulder"]),
        (nj["r_shoulder"], nj["r_hip"]),
        (nj["l_hip"], nj["r_hip"]),
        # left f leg
        (nj["l_shoulder"], nj["l_elbow"]),
        (nj["l_elbow"], nj["l_wrist"]),
        (nj["l_wrist"], nj["lf_paw"]),
        # right f leg
        (nj["r_shoulder"], nj["r_elbow"]),
        (nj["r_elbow"], nj["r_wrist"]),
        (nj["r_wrist"], nj["rf_paw"]),
        # left back leg
        (nj["l_hip"], nj["l_knee"]),
        (nj["l_knee"], nj["l_ankle"]),
        (nj["l_ankle"], nj["lb_paw"]),
        # right back leg
        (nj["r_hip"], nj["r_knee"]),
        (nj["r_knee"], nj["r_ankle"]),
        (nj["r_ankle"], nj["rb_paw"]),
    ]


def rotate_by(vec: np.ndarray, xz_angle: np.ndarray):
    rotation = Rotation.from_euler("xz", xz_angle, degrees=True)
    return rotation.apply(vec)


def rotate_z_by(vec, angle: float):
    rotation = Rotation.from_euler("z", angle, degrees=True)
    return rotation.apply(vec)


def unpack_dog_scalar_params(shape_scalar_params: np.ndarray):
    neck_scalar = shape_scalar_params[0]
    nose_head_scalar = shape_scalar_params[1]
    shoulder_elbow_scalar = shape_scalar_params[2]
    elbow_wrist_scalar = shape_scalar_params[3]
    wrist_paw_scalar = shape_scalar_params[4]
    hip_knee_scalar = shape_scalar_params[5]
    knee_ankle_scalar = shape_scalar_params[6]
    ankle_paw_scalar = shape_scalar_params[7]
    return (
        neck_scalar,
        nose_head_scalar,
        shoulder_elbow_scalar,
        elbow_wrist_scalar,
        wrist_paw_scalar,
        hip_knee_scalar,
        knee_ankle_scalar,
        ankle_paw_scalar,
    )


def unpack_dog_angle_params(
    shape_angle_params: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
]:
    neck_angle = shape_angle_params[0:2]
    nose_head_angle = shape_angle_params[2:4]
    l_shoulder_elbow_angle = shape_angle_params[4:6]
    r_shoulder_elbow_angle = shape_angle_params[6:8]
    l_elbow_wrist_angle = shape_angle_params[8]
    r_elbow_wrist_angle = shape_angle_params[9]
    l_wrist_paw_angle = shape_angle_params[10]
    r_wrist_paw_angle = shape_angle_params[11]
    l_hip_knee_angle = shape_angle_params[12:14]
    r_hip_knee_angle = shape_angle_params[14:16]
    l_knee_ankle_angle = shape_angle_params[16]
    r_knee_ankle_angle = shape_angle_params[17]
    l_ankle_paw_angle = shape_angle_params[18]
    r_ankle_paw_angle = shape_angle_params[19]

    return (
        neck_angle,
        nose_head_angle,
        l_shoulder_elbow_angle,
        r_shoulder_elbow_angle,
        l_elbow_wrist_angle,
        r_elbow_wrist_angle,
        l_wrist_paw_angle,
        r_wrist_paw_angle,
        l_hip_knee_angle,
        r_hip_knee_angle,
        l_knee_ankle_angle,
        r_knee_ankle_angle,
        l_ankle_paw_angle,
        r_ankle_paw_angle,
    )


def _construct_model_joints(
    shape_scalar_params: np.ndarray, shape_angle_params: np.ndarray
) -> list[np.ndarray]:

    (
        neck_scalar,
        nose_head_scalar,
        shoulder_elbow_scalar,
        elbow_wrist_scalar,
        wrist_paw_scalar,
        hip_knee_scalar,
        knee_ankle_scalar,
        ankle_paw_scalar,
    ) = unpack_dog_scalar_params(shape_scalar_params)

    (
        neck_angle,
        nose_head_angle,
        l_shoulder_elbow_angle,
        r_shoulder_elbow_angle,
        l_elbow_wrist_angle,
        r_elbow_wrist_angle,
        l_wrist_paw_angle,
        r_wrist_paw_angle,
        l_hip_knee_angle,
        r_hip_knee_angle,
        l_knee_ankle_angle,
        r_knee_ankle_angle,
        l_ankle_paw_angle,
        r_ankle_paw_angle,
    ) = unpack_dog_angle_params(shape_angle_params)

    neck = np.array([125, 115, 0], dtype="float64")
    half_width = 32
    r_shoulder = np.array([132, 161, -half_width], dtype="float64")
    l_shoulder = np.array([132, 161, half_width], dtype="float64")

    r_hip = np.array([323, 152, -half_width], dtype="float64")
    l_hip = np.array([323, 152, half_width], dtype="float64")

    neck_length = 60 * neck_scalar
    nose_head_length = 50 * nose_head_scalar
    neck_dir = rotate_by(np.array([-1, 0, 0]), neck_angle)
    nose_dir = rotate_by(np.array([-1, 0, 0]), neck_angle + nose_head_angle)
    head = neck + neck_dir * neck_length
    nose = head + nose_dir * nose_head_length

    shoulder_elbow_length = 80 * shoulder_elbow_scalar
    elbow_wrist_length = 80 * elbow_wrist_scalar
    wrist_paw_length = 40 * wrist_paw_scalar
    hip_knee_length = 80 * hip_knee_scalar
    knee_ankle_length = 80 * knee_ankle_scalar
    ankle_paw_length = 40 * ankle_paw_scalar

    shoulder_elbow_dir = np.array([0, 1, 0])
    elbow_wrist_dir = np.array([0, 1, 0])
    wrist_paw_dir = np.array([0, 1, 0])
    hip_knee_dir = np.array([0, 1, 0])
    knee_ankle_dir = np.array([0, 1, 0])
    ankle_paw_dir = np.array([0, 1, 0])

    l_elbow = l_shoulder + shoulder_elbow_length * rotate_by(
        shoulder_elbow_dir, l_shoulder_elbow_angle
    )
    r_elbow = r_shoulder + shoulder_elbow_length * rotate_by(
        shoulder_elbow_dir, r_shoulder_elbow_angle
    )

    l_wrist = l_elbow + elbow_wrist_length * rotate_z_by(
        elbow_wrist_dir, l_shoulder_elbow_angle[1] + l_elbow_wrist_angle
    )
    r_wrist = r_elbow + elbow_wrist_length * rotate_z_by(
        elbow_wrist_dir, r_shoulder_elbow_angle[1] + r_elbow_wrist_angle
    )

    lf_paw = l_wrist + wrist_paw_length * rotate_z_by(
        wrist_paw_dir,
        l_shoulder_elbow_angle[1] + l_elbow_wrist_angle + l_wrist_paw_angle,
    )
    rf_paw = r_wrist + wrist_paw_length * rotate_z_by(
        wrist_paw_dir,
        r_shoulder_elbow_angle[1] + r_elbow_wrist_angle + r_wrist_paw_angle,
    )

    l_knee = l_hip + hip_knee_length * rotate_by(hip_knee_dir, l_hip_knee_angle)
    r_knee = r_hip + hip_knee_length * rotate_by(hip_knee_dir, r_hip_knee_angle)

    l_ankle = l_knee + knee_ankle_length * rotate_z_by(
        knee_ankle_dir, l_hip_knee_angle[1] + l_knee_ankle_angle
    )
    r_ankle = r_knee + knee_ankle_length * rotate_z_by(
        knee_ankle_dir, r_hip_knee_angle[1] + r_knee_ankle_angle
    )
    lb_paw = l_ankle + ankle_paw_length * rotate_z_by(
        ankle_paw_dir, l_hip_knee_angle[1] + l_knee_ankle_angle + l_ankle_paw_angle
    )
    rb_paw = r_ankle + ankle_paw_length * rotate_z_by(
        ankle_paw_dir, r_hip_knee_angle[1] + r_knee_ankle_angle + r_ankle_paw_angle
    )

    joints = [
        nose,
        head,
        neck,
        l_shoulder,
        l_elbow,
        l_wrist,
        lf_paw,
        r_shoulder,
        r_elbow,
        r_wrist,
        rf_paw,
        l_hip,
        l_knee,
        l_ankle,
        lb_paw,
        r_hip,
        r_knee,
        r_ankle,
        rb_paw,
    ]
    return joints


def _scale_joints(joints3d: list[np.ndarray], scale: float):
    return [joint * scale for joint in joints3d]


def _rotate3d_joints(joints3d: list[np.ndarray], rot: np.ndarray):
    r = Rotation.from_euler("zyx", rot, degrees=True)
    return [r.apply(joint) for joint in joints3d]


def _project_plane(joints: list[np.ndarray]):
    return [joint[:2] for joint in joints]


def _shift2d_joints(joints2d: list[np.ndarray], shift: np.ndarray):
    return [joint + shift for joint in joints2d]


def _apply_transform(joints, scale, rot, shift) -> list[np.ndarray]:
    scaled_joints = _scale_joints(joints, scale)
    rotated_joints = _rotate3d_joints(scaled_joints, rot)
    projected_joints = _project_plane(rotated_joints)
    d2_joints = _shift2d_joints(projected_joints, shift)
    return d2_joints


def _unpack_params(params):
    scale = params[0]
    rot = params[1:4]
    shift = params[4:6]
    shape_scalars_params = params[6:14]
    shape_angle_params = params[14:]
    return scale, rot, shift, shape_scalars_params, shape_angle_params


def _pack_params(
    scale: float,
    rot: np.ndarray,
    shift: np.ndarray,
    shape_scalars_params: np.ndarray,
    shape_angle_params: np.ndarray,
) -> np.ndarray:
    return np.array(
        [scale, *rot, *shift, *shape_scalars_params, *shape_angle_params],
        dtype="float64",
    )


def _loss_function_dog_frame(params: np.ndarray, d2_keypoints: list[np.ndarray]):
    scale, rot, shift = params[0], params[1:4], params[4:6]
    joints: list[np.ndarray] = _construct_model_joints(
        np.zeros(8, dtype="float64"), np.zeros(20, dtype="float64")
    )
    d2_joints = _apply_transform(joints, scale, rot, shift)
    # punish much harder for head and frame
    loss = 0
    for i in [0, 1, 2, 3, 7, 11, 15]:
        joint, keypoint = d2_joints[i], d2_keypoints[i]
        loss += linalg.norm(joint - keypoint)
    return loss


def general_loss_function(params: np.ndarray, d2_keypoints: list[np.ndarray]):
    scale, rot, shift, shape_scalars_params, shape_angle_params = _unpack_params(params)
    joints: list[np.ndarray] = _construct_model_joints(
        shape_scalars_params, shape_angle_params
    )
    d2_joints = _apply_transform(joints, scale, rot, shift)

    loss = sum(
        linalg.norm(joint - keypoint)
        for joint, keypoint in zip(d2_joints, d2_keypoints)
    )
    # punish much harder for head and frame
    for i in [0, 1, 2, 3, 7, 11, 15]:
        joint, keypoint = d2_joints[i], d2_keypoints[i]
        loss += linalg.norm(joint - keypoint) ** 2
    return loss


def legs_loss_function(
    shape_angle_params: np.ndarray,
    scale,
    rot,
    shift,
    shape_scalars_params,
    d2_keypoints: list[np.ndarray],
):
    joints: list[np.ndarray] = _construct_model_joints(
        shape_scalars_params, shape_angle_params
    )
    d2_joints = _apply_transform(joints, scale, rot, shift)

    loss = sum(
        linalg.norm(joint - keypoint)
        for joint, keypoint in zip(d2_joints, d2_keypoints)
    )
    # punish much harder for head and frame
    for i in [0, 1, 2, 3, 7, 11, 15]:
        joint, keypoint = d2_joints[i], d2_keypoints[i]
        loss += linalg.norm(joint - keypoint) ** 2
    return loss


def optimize(
    d2_keypoints: list[np.ndarray],
):
    # first opti for frame
    frame_guess = minimize(
        _loss_function_dog_frame,
        x0=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # scale  # rotation  # shift
        args=(d2_keypoints),
        options={"disp": True},
    )

    yield (10, None)
    dog_shape_init_guess_params = [
        1.0,  # scalars 1
        1.0,  # 2
        1.0,  # 3
        1.0,  # 4
        1.0,  # 5
        1.0,  # 6
        1.0,  # 7
        1.0,  # 8
        0.0,  # angles
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    bounds = [
        (0, None),  # scale
        (None, None),  # rotation x
        (None, None),  # y
        (None, None),  # z
        (None, None),  # shift x
        (None, None),  # y
        (0.96, 1.05),  # scalars for lenght
        (0.96, 1.05),
        (0.96, 1.05),
        (0.96, 1.05),
        (0.96, 1.05),
        (0.96, 1.05),
        (0.96, 1.05),
        (0.96, 1.05),
        (-45, 45),  # neck
        (-45, 45),
        (-45, 45),  # head
        (-45, 45),
        # shoulder_elbow_angle
        (-20, 20),
        (-45, 80),
        (-20, 20),
        (-45, 80),
        # elbow_wrist_angle
        (0, 80),
        (0, 80),
        # wrist_paw_angle
        (-80, 10),
        (-80, 10),
        # hip_knee_angle
        (-20, 20),
        (-80, 80),
        (-20, 20),
        (-80, 80),
        # knee_ankle_angle
        (-80, 0),
        (-80, 0),
        # ankle_paw_angle
        (0, 80),
        (0, 80),
    ]

    basic = frame_guess.x.copy()
    # dog is often flipped upside down
    x_rot = frame_guess.x.copy()
    x_rot[0] += 180
    # dog is often flipped with the wrong side pointing to the camera
    y_rot = frame_guess.x.copy()
    y_rot[1] += 180
    # if its flipped badly on both these axes
    z_rot = frame_guess.x.copy()
    z_rot[2] += 180
    frameres_init_guess = [basic, y_rot, x_rot, z_rot]

    # find best optimisation
    best_result = None
    best_loss = 10000000000
    for i, init_guess in enumerate(frameres_init_guess):
        general_guess = minimize(
            general_loss_function,
            x0=[*init_guess, *dog_shape_init_guess_params],
            args=(d2_keypoints),
            bounds=bounds,
            options={"disp": True},
        )
        if not best_result or general_guess.fun < best_loss:
            best_loss = general_guess.fun
            best_result = general_guess
        yield (10 + (i + 1) * 20, None)

    assert best_result is not None
    scale, rot, shift, shape_scalars_params, shape_angle_params = _unpack_params(
        best_result.x
    )
    _, _, _, _, shape_angle_bounds = _unpack_params(bounds)

    legs_guess = minimize(
        legs_loss_function,
        x0=shape_angle_params,
        args=(scale, rot, shift, shape_scalars_params, d2_keypoints),
        bounds=shape_angle_bounds,
        options={"disp": True},
    )

    print(legs_guess.x)
    print(_pack_params(scale, rot, shift, shape_scalars_params, legs_guess.x))

    yield (100, _pack_params(scale, rot, shift, shape_scalars_params, legs_guess.x))
