#############
# body pose #
#############

struct Point3
    x::Float64
    y::Float64
    z::Float64
end

Point3(tup::Tuple{U,U,U}) where {U<:Real} = Point3(tup[1], tup[2], tup[3])

Base.:+(a::Point3, b::Point3) = Point3(a.x + b.x, a.y + b.y, a.z + b.z)
Base.:-(a::Point3, b::Point3) = Point3(a.x - b.x, a.y - b.y, a.z - b.z)
Base.norm(a::Point3) = sqrt(a.x * a.x + a.y * a.y + a.z * a.z)

tup(point::Point3) = (point.x, point.y, point.z)

struct BodyPose
    rotation::Point3
    elbow_r_loc::Point3
    elbow_l_loc::Point3
    elbow_r_rot::Point3
    elbow_l_rot::Point3
    hip_loc::Point3
    heel_r_loc::Point3
    heel_l_loc::Point3
end

function BodyPose(choices::ChoiceTrie)
    rotation_x = choices[:rotation]
    rotation = scale_rot(rotation_x)
    elbow_r_loc_x = choices[:elbow_r_loc_x]
    elbow_r_loc_y = choices[:elbow_r_loc_y]
    elbow_r_loc_z = choices[:elbow_r_loc_z]
    elbow_r_loc = scale_elbow_r_loc(elbow_r_loc_x, elbow_r_loc_y, elbow_r_loc_z)
    elbow_l_loc_x = choices[:elbow_l_loc_x]
    elbow_l_loc_y = choices[:elbow_l_loc_y]
    elbow_l_loc_z = choices[:elbow_l_loc_z]
    elbow_l_loc = scale_elbow_l_loc(elbow_l_loc_x, elbow_l_loc_y, elbow_l_loc_z)
    elbow_r_rot_z = choices[:elbow_r_rot_z]
    elbow_r_rot = scale_elbow_r_rot(elbow_r_rot_z)
    elbow_l_rot_z = choices[:elbow_l_rot_z]
    elbow_l_rot = scale_elbow_l_rot(elbow_l_rot_z)
    hip_loc_z = choices[:hip_loc_z]
    hip_loc = scale_hip_loc(hip_loc_z)
    heel_r_loc_x = choices[:heel_r_loc_x]
    heel_r_loc_y = choices[:heel_r_loc_y]
    heel_r_loc_z = choices[:heel_r_loc_z]
    heel_r_loc = scale_heel_r_loc(heel_r_loc_x, heel_r_loc_y, heel_r_loc_z)
    heel_l_loc_x = choices[:heel_l_loc_x]
    heel_l_loc_y = choices[:heel_l_loc_y]
    heel_l_loc_z = choices[:heel_l_loc_z]
    heel_l_loc = scale_heel_l_loc(heel_l_loc_x, heel_l_loc_y, heel_l_loc_z)
    return BodyPose(
        rotation,
        elbow_r_loc,
        elbow_l_loc,
        elbow_r_rot,
        elbow_l_rot,
        hip_loc,
        heel_r_loc,
        heel_l_loc)
end

function Base.:+(a::BodyPose, b::BodyPose)
    BodyPose(
        a.rotation + b.rotation,
        a.elbow_r_loc + b.elbow_r_loc,
        a.elbow_l_loc + b.elbow_l_loc,
        a.elbow_r_rot + b.elbow_r_rot,
        a.elbow_l_rot + b.elbow_l_rot,
        a.hip_loc + b.hip_loc,
        a.heel_r_loc + b.heel_r_loc,
        a.heel_l_loc + b.heel_l_loc)
end

function add_columns!(pose::BodyPose, df::DataFrame)
    df[:rotation_x] = pose.rotation.x
    df[:rotation_y] = pose.rotation.y
    df[:rotation_z] = pose.rotation.z
    df[:elbow_r_loc_x] = pose.elbow_r_loc.x
    df[:elbow_r_loc_y] = pose.elbow_r_loc.y
    df[:elbow_r_loc_z] = pose.elbow_r_loc.z
    df[:elbow_l_loc_x] = pose.elbow_l_loc.x
    df[:elbow_l_loc_y] = pose.elbow_l_loc.y
    df[:elbow_l_loc_z] = pose.elbow_l_loc.z
    df[:elbow_l_rot_z] = pose.elbow_l_rot.z
    df[:elbow_r_rot_z] = pose.elbow_r_rot.z
    df[:hip_loc_x] = pose.hip_loc.x
    df[:hip_loc_y] = pose.hip_loc.y
    df[:hip_loc_z] = pose.hip_loc.z
    df[:heel_r_loc_x] = pose.heel_r_loc.x
    df[:heel_r_loc_y] = pose.heel_r_loc.y
    df[:heel_r_loc_z] = pose.heel_r_loc.z
    df[:heel_l_loc_x] = pose.heel_l_loc.x
    df[:heel_l_loc_y] = pose.heel_l_loc.y
    df[:heel_l_loc_z] = pose.heel_l_loc.z
end

function square_error(pose1::BodyPose, pose2::BodyPose)
    err = 0.
    err += norm(pose1.rotation - pose2.rotation)^2
    err += norm(pose1.elbow_r_loc - pose2.elbow_r_loc)^2
    err += norm(pose1.elbow_l_loc - pose2.elbow_l_loc)^2
    err += norm(pose1.elbow_r_rot - pose2.elbow_r_rot)^2
    err += norm(pose1.elbow_l_rot - pose2.elbow_l_rot)^2
    err += norm(pose1.hip_loc - pose2.hip_loc)^2
    err += norm(pose1.heel_r_loc - pose2.heel_r_loc)^2
    err += norm(pose1.heel_l_loc - pose2.heel_l_loc)^2
    return err
end

###############
# scene model #
###############

# rescale values from [0, 1] to another interval
scale(value, min, max) = min + (max - min) * value
scale_rot(z) = Point3(0., 0., scale(z, -pi/4, pi/4))
scale_elbow_r_loc(x, y, z) = Point3(scale(x, -1, 0), scale(y, -1, 1), scale(z, -1, 1))
scale_elbow_r_rot(z) = Point3(0., 0., scale(z, 0, 2*pi))
scale_elbow_l_loc(x, y, z) = Point3(scale(x, 0, 1), scale(y, -1, 1), scale(z, -1, 1))
scale_elbow_l_rot(z) = Point3(0., 0., scale(z, 0, 2*pi))
scale_hip_loc(z) = Point3(0., 0., scale(z, -0.35, 0))
scale_heel_r_loc(x, y, z) = Point3(scale(x, -0.45, 0.1), scale(y, -1, 0.5), scale(z, -0.2, 0.2))
scale_heel_l_loc(x, y, z) = Point3(scale(x, -0.1, 0.45), scale(y, -1, 0.5), scale(z, -0.2, 0.2))

@compiled @gen function body_pose_model()

    # global rotation
    rotation_x::Float64 = @addr(uniform(0, 1), :rotation)
    rotation::Point3 = scale_rot(rotation_x)

    # right elbow location
    elbow_r_loc_x::Float64 = @addr(uniform(0, 1), :elbow_r_loc_x)
    elbow_r_loc_y::Float64 = @addr(uniform(0, 1), :elbow_r_loc_y)
    elbow_r_loc_z::Float64 = @addr(uniform(0, 1), :elbow_r_loc_z)
    elbow_r_loc::Point3 = scale_elbow_r_loc(elbow_r_loc_x, elbow_r_loc_y, elbow_r_loc_z)
    
    # left elbow location
    elbow_l_loc_x::Float64 = @addr(uniform(0, 1), :elbow_l_loc_x)
    elbow_l_loc_y::Float64 = @addr(uniform(0, 1), :elbow_l_loc_y)
    elbow_l_loc_z::Float64 = @addr(uniform(0, 1), :elbow_l_loc_z)
    elbow_l_loc::Point3 = scale_elbow_l_loc(elbow_l_loc_x, elbow_l_loc_y, elbow_l_loc_z)

    # right elbow rotation
    elbow_r_rot_z::Float64 = @addr(uniform(0, 1), :elbow_r_rot_z)
    elbow_r_rot::Point3 = scale_elbow_r_rot(elbow_r_rot_z)

    # left elbow rotation
    elbow_l_rot_z::Float64 = @addr(uniform(0, 1), :elbow_l_rot_z)
    elbow_l_rot::Point3 = scale_elbow_l_rot(elbow_l_rot_z)

    # hip
    hip_loc_z::Float64 = @addr(uniform(0, 1), :hip_loc_z)
    hip_loc::Point3 = scale_hip_loc(hip_loc_z)

    # right heel
    heel_r_loc_x::Float64 = @addr(uniform(0, 1), :heel_r_loc_x)
    heel_r_loc_y::Float64 = @addr(uniform(0, 1), :heel_r_loc_y)
    heel_r_loc_z::Float64 = @addr(uniform(0, 1), :heel_r_loc_z)
    heel_r_loc::Point3 = scale_heel_r_loc(heel_r_loc_x, heel_r_loc_y, heel_r_loc_z)

    # left heel
    heel_l_loc_x::Float64 = @addr(uniform(0, 1), :heel_l_loc_x)
    heel_l_loc_y::Float64 = @addr(uniform(0, 1), :heel_l_loc_y)
    heel_l_loc_z::Float64 = @addr(uniform(0, 1), :heel_l_loc_z)
    heel_l_loc::Point3 = scale_heel_l_loc(heel_l_loc_x, heel_l_loc_y, heel_l_loc_z)

    return BodyPose(
        rotation,
        elbow_r_loc,
        elbow_l_loc,
        elbow_r_rot,
        elbow_l_rot,
        hip_loc,
        heel_r_loc,
        heel_l_loc)::BodyPose
end

struct BodyPoseSceneModel end

function sample(::BodyPoseSceneModel)
    trace = simulate(body_pose_model, ())
    return get_call_record(trace).retval::BodyPose
end
