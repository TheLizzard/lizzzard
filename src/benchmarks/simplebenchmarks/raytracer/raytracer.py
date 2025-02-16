import math

math._tan = math.tan
math._sin = math.sin
math._cos = math.cos
math.tan = lambda x: math._tan(x*math.pi/180)
math.sin = lambda x: math._sin(x*math.pi/180)
math.cos = lambda x: math._cos(x*math.pi/180)
_isinstance = isinstance
def isinstance(a, b):
    if b is float:
        b = float|int
    return _isinstance(a, b)


ε = 0.000001
clamp = lambda val, low, high: low if val < low else (high if val > high else val)
float_abs = lambda val: val if val > 0 else -val
min = lambda x, y: x if x < y else y
max = lambda x, y: x if x > y else y
def _assert(boolean, error):
    if _isinstance(bool, float):
        boolean = not (-ε < boolean < ε)
    if (not boolean):
        io.print(error)
        io.print(1/0)
float_random = lambda: 0


############################################################# Vector Colour Line
class Vector:
    def __init__(this, x, y, z):
        _assert(isinstance(x, float), "TypeError")
        _assert(isinstance(y, float), "TypeError")
        _assert(isinstance(z, float), "TypeError")
        this.x = x
        this.y = y
        this.z = z

    add = lambda this, other: Vector(this.x+other.x, this.y+other.y, this.z+other.z)
    sub = lambda this, other: this.add(other.negate())
    dot = lambda this, other: this.x*other.x + this.y*other.y + this.z*other.z
    cross = lambda this, other: Vector(this.y*other.z-this.z*other.y, -(this.x*other.z-this.z*other.x), this.x*other.y-this.y*other.x)
    mul2 = lambda this, other: Vector(this.x*other.x, this.y*other.y, this.z*other.z)

    repr = lambda this: f"Vector[{this.x:.4f}, {this.y:.4f}, {this.z:.4f}]"
    negate = lambda this: Vector(-this.x, -this.y, -this.z)

    mul = lambda this, scalar: Vector(this.x*scalar, this.y*scalar, this.z*scalar)
    div = lambda this, scalar: this.mul(1/scalar)

    abs2 = lambda this: this.dot(this)
    dist2 = lambda this, other: this.sub(other).abs2()
    eq = lambda this, other: this.dist2(other) < ε

    isunit = lambda this: 1-ε < this.abs2() < 1+ε
    unit = lambda this: this.div(math.sqrt(this.abs2()))
    normalise = unit

    clamp = lambda this, low, high: Vector(clamp(this.x, low, high), clamp(this.y, low, high), clamp(this.z, low, high))


class Line:
    def __init__(this, start, dir):
        _assert(isinstance(start, Vector), "TypeError")
        _assert(isinstance(dir, Vector), "TypeError")
        _assert(dir.isunit(), "InternalError")
        this.start = start
        this.dir = dir

    from_points = lambda start, end: Line(start, end.sub(start).unit())


class Colour:
    def __init__(this, data):
        _assert(isinstance(data, Vector), "TypeError")
        this.data = data

    add = lambda this, other: Colour(this.data.add(other.data))
    mul2 = lambda this, other: Colour(this.data.mul2(other.data))
    mul = lambda this, other: Colour(this.data.mul(other))
    div = lambda this, other: Colour(this.data.div(other))
    clamp = lambda this, low, high: Colour(this.data.clamp(low, high))
    repr = lambda this: "Colour" + this.data.repr()[6:]
    r = lambda this: this.data.x
    g = lambda this: this.data.y
    b = lambda this: this.data.z


####################################################################### Material
class Material:
    def __init__(this, ks, kd, specularexponent, diffusecolour, specularcolour, isreflective, reflectivity, isrefractive, refractiveindex):
        this.ks = ks
        this.kd = kd
        this.specularexponent = specularexponent
        this.diffusecolour = diffusecolour
        this.specularcolour = specularcolour
        this.isreflective = isreflective
        this.reflectivity = reflectivity
        this.isrefractive = isrefractive
        this.refractiveindex = refractiveindex


######################################################################### Object
class Object:
    def __init__(this, material):
        _assert(isinstance(material, Material), "TypeError")
        this.material = material

class Triangle(Object):
    def __init__(this, material, v1, v2, v3):
        _assert(isinstance(v1, Vector), "TypeError")
        _assert(isinstance(v2, Vector), "TypeError")
        _assert(isinstance(v3, Vector), "TypeError")
        Object.__init__(this, material)
        this.v1 = v1
        this.v2 = v2
        this.v3 = v3

    def intersects_line(this, line):
        _assert(isinstance(line, Line), "TypeError")
        # Wikipedia: Möller–Trumbore intersection algorithm
        edge1 = this.v2.sub(this.v1)
        edge2 = this.v3.sub(this.v1)
        ray_cross_e2 = line.dir.cross(edge2)
        det = edge1.dot(ray_cross_e2)
        if -ε < det < ε:
            return NO_INTERSECTION
        inv_det = 1 / det
        s = line.start.sub(this.v1)
        u = inv_det * s.dot(ray_cross_e2)
        if (((u < 0) and (float_abs(u) > ε)) or ((u > 1) and (float_abs(u-1) > ε))):
            return NO_INTERSECTION
        s_cross_e1 = s.cross(edge1)
        v = inv_det * line.dir.dot(s_cross_e1)
        if (((v < 0) and (float_abs(v) > ε)) or ((u + v > 1) and (float_abs(u + v - 1) > ε))):
            return NO_INTERSECTION
        λ = inv_det * edge2.dot(s_cross_e1)
        if (λ < ε):
            return NO_INTERSECTION
        return Intersection(this, line.start.add(line.dir.mul(λ)))

    def normal(this, point):
        _assert(isinstance(point, Vector), "TypeError")
        edge1 = this.v2.sub(this.v1)
        edge2 = this.v3.sub(this.v1)
        return edge1.cross(edge2).unit()

class Sphere(Object):
    def __init__(this, material, center, r):
        _assert(isinstance(center, Vector), "TypeError")
        _assert(isinstance(r, float), "TypeError")
        Object.__init__(this, material)
        this.center = center
        this.r = r

    def intersects_line(this, line):
        _assert(isinstance(line, Line), "TypeError")
        intersection = this.line_inside_sphere(line)
        if (intersection == NO_INTERSECTION):
            return NO_INTERSECTION
        λ = intersection.where.sub(line.start).dot(line.dir)
        if (λ < ε):
            return NO_INTERSECTION
        debug = line.start.add(line.dir.mul(λ))
        _assert(debug.eq(intersection.where), "InternalError")
        return intersection

    def line_inside_sphere(this, line):
        _assert(isinstance(line, Line), "TypeError")
        # Computed this by hand
        n = line.start.sub(this.center)
        tmp = -line.dir.dot(n)
        Δ = tmp*tmp - n.abs2() + this.r*this.r
        if (Δ < 0):
            return NO_INTERSECTION
        sqrtΔ = math.sqrt(Δ)
        λ = min(tmp-sqrtΔ, tmp+sqrtΔ)
        return Intersection(this, line.dir.mul(λ).add(line.start))

    def normal(this, point):
        return point.sub(this.center).unit()

class Cylinder(Object):
    def __init__(this, material, top_dir, center, height, r):
        _assert(isinstance(top_dir, Vector), "TypeError")
        _assert(isinstance(height, float), "TypeError")
        _assert(isinstance(center, Vector), "TypeError")
        _assert(isinstance(r, float), "TypeError")
        _assert(top_dir.isunit(), "InternalError")
        base = center.add(top_dir.mul(-height))
        Object.__init__(this, material)
        this.top_dir = top_dir
        this.base = base
        this.height = height*2
        this.r = r

    def intersects_line(this, line):
        _assert(isinstance(line, Line), "TypeError")
        # https://en.wikipedia.org/wiki/Line-cylinder_intersection
        base = this.base.sub(line.start)
        tmp = line.dir.cross(this.top_dir)
        if -ε < tmp.abs2() < ε:
            # Cylinder parallel to viewer so no intersections
            return NO_INTERSECTION
        tmp2 = base.dot(tmp)
        tmp3 = tmp.abs2()
        Δ = tmp3*this.r*this.r - tmp2*tmp2
        if (Δ < 0):
            return NO_INTERSECTION
        neg_b = tmp.dot(base.cross(this.top_dir))
        sqrtΔ = math.sqrt(Δ)
        λ = min(neg_b-sqrtΔ, neg_b+sqrtΔ) / tmp3
        intersect_point = line.start.add(line.dir.mul(λ))
        tmp2 = intersect_point.sub(this.base).dot(this.top_dir)
        if ((tmp2 < 0) or (tmp2 > this.height)):
            return NO_INTERSECTION
        if (λ < ε):
            return NO_INTERSECTION
        return Intersection(this, intersect_point)

    def normal(this, point):
        _assert(isinstance(point, Vector), "TypeError")
        tmp = point.sub(this.base)
        tmp2 = tmp.dot(this.top_dir)
        raw_normal = tmp.sub(this.top_dir.mul(tmp2))
        # normal = raw_normal.div(this.r) # Inacurate
        return raw_normal.unit()


################################################################### Intersection
class Intersection:
    def __init__(this, what, where):
        if where is not None:
            _assert(isinstance(where, Vector), "TypeError")
        if what is not None:
            _assert(isinstance(what, Object), "TypeError")
        this.where = where
        this.what = what


def line_intersections(ray, scene):
    _assert(isinstance(scene, Scene), "TypeError")
    _assert(isinstance(ray, Line), "TypeError")
    i = -1
    output = []
    while True:
        i += 1
        if (i >= len(scene.objs)):
            break
        obj = scene.objs[i]
        intersection = obj.intersects_line(ray)
        if (intersection.what):
            output.append(intersection)
    return output


######################################################## Camera PointLight Scene
class Camera:
    def __init__(this, width, height, looking_dir, upwards_dir, pos, fov, exposure):
        this.width = width
        this.height = height
        this.looking_dir = looking_dir.unit()
        this.upwards_dir = upwards_dir.unit()
        this.pos = pos
        this.fov = fov
        this.exposure = exposure

class PointLight:
    def __init__(this, colour, pos):
        assert isinstance(colour, Colour), "TypeError"
        this.colour = colour
        this.pos = pos

class Scene:
    def __init__(this, point_lights, objs, camera, bg, nbounces):
        _assert(isinstance(point_lights, list), "TypeError")
        _assert(isinstance(camera, Camera), "TypeError")
        _assert(isinstance(nbounces, int), "TypeError")
        _assert(isinstance(objs, list), "TypeError")
        _assert(isinstance(bg, Colour), "TypeError")
        this.point_lights = point_lights
        this.objs = objs
        this.camera = camera
        this.bg = bg
        this.nbounces = nbounces


###################################################################### RayTracer
def save_img(image, width, height, path):
    u8 = lambda x: int(clamp(x+0.5, 0, 255))
    file = open(path, "w")
    file.write("P3\n")
    file.write("# :-)\n")
    file.write(str(width) + " " + str(height) + "\n")
    file.write("255\n")
    y = -1
    while True:
        y += 1
        if y >= height:
            break
        x = -1
        while True:
            x += 1
            if x >= width:
                break
            colour = image[y*width+x]
            file.write(str(u8(255*colour.r())))
            file.write(" ")
            file.write(str(u8(255*colour.g())))
            file.write(" ")
            file.write(str(u8(255*colour.b())))
            file.write(" ")
        file.write("\n")
    file.close()

tone_map = lambda image, width, height, camera: None

reflect = lambda dir, normal: dir.sub(normal.mul(2*dir.dot(normal))).unit()

def refract(dir, normal, eta):
    cosi = -dir.dot(normal)
    sint2 = eta * eta * (1 - cosi*cosi)
    if (sint2 > 1):
        return dir.negate() # Total internal reflection
    cost = math.sqrt(1 - sint2)
    return dir.mul(eta).add(normal.mul(eta*cosi-cost)).unit()

def compute_lighting(point, normal, view, material, scene):
    _assert(normal.isunit(), "InternalError")
    _assert(view.isunit(), "InternalError")

    result = scene.bg.mul(scene.camera.exposure)

    i = -1
    while True:
        i += 1
        if (i >= len(scene.point_lights)):
            break
        light = scene.point_lights[i]
        # Calculate the direction to the light source
        light_dir = light.pos.sub(point)
        distance_to_light = light_dir.abs2()
        light_dir = light_dir.div(math.sqrt(distance_to_light))
        _assert(light_dir.isunit(), "InternalError")

        # Shadow check: cast a ray from the point to the light
        new_point = point.add(light_dir.mul(1/1000))
        shadow_ray = Line(new_point, light_dir)
        in_shadow = False

        shadow_intersections = line_intersections(shadow_ray, scene)
        j = -1
        while True:
            j += 1
            if (j >= len(shadow_intersections)):
                break
            shadow_intersection = shadow_intersections[j]
            dist = shadow_intersection.where.sub(point).abs2()
            if (dist < distance_to_light):
                in_shadow = True
                break
        if (in_shadow): continue

        # Diffuse lighting
        diff_intensity = light_dir.dot(normal)
        if (material and (diff_intensity > 0)):
            diffl = light.colour.mul2(material.diffusecolour).mul(diff_intensity*material.kd)
            result = result.add(diffl)

        # Specular lighting
        if (material and (material.ks > 0)):
            reflected = reflect(light_dir.mul(-1), normal)
            spec_intensity = reflected.dot(view)
            if (spec_intensity > 0):
                spec_intensity = math.pow(spec_intensity, material.specularexponent)
                spec = light.colour.mul2(material.specularcolour).mul(spec_intensity*material.ks)
                result = result.add(spec)

    # Clamp the result colour values to [0, 1]
    return result.clamp(0, 1)


def trace_ray(ray, scene, depth):
    if (depth > scene.nbounces):
        return scene.bg

    # Initialize variables to store the closest intersection
    closest_intersection = NO_INTERSECTION
    closest_object = None
    closest_dist = 0

    # Find the closest intersection point
    intersections = line_intersections(ray, scene)
    i = -1
    while True:
        i += 1
        if (i >= len(intersections)):
            break
        intersection = intersections[i]
        dist = intersection.where.sub(ray.start).dot(ray.dir)
        # If the object is behind us
        if (dist < 0):
            continue
        # If this is a closer interaction
        if ((not closest_intersection.what) or (dist < closest_dist)):
            closest_intersection = intersection
            closest_object = intersection.what
            closest_dist = dist

    # If no intersection is found, return the background colour
    if (not closest_intersection.what):
        return scene.bg

    # Retrieve material and compute the normal at the intersection point
    material = closest_object.material
    point = closest_intersection.where
    normal = closest_object.normal(point)
    _assert(normal.isunit(), "InternalError")

    # Adjust the normal direction to ensure it's facing against the ray direction
    if (normal.dot(ray.dir) > 0):
        normal = normal.negate()

    # Compute lighting at the intersection point
    colour = compute_lighting(point, normal, ray.dir.negate(), material, scene)

    # Reflection handling
    if (material and material.isreflective and (depth < scene.nbounces)):
        reflected_ray = Line(point, reflect(ray.dir, normal))
        # Trace the reflected ray
        reflected_colour = trace_ray(reflected_ray, scene, depth+1)
        # Blend the reflected colour with the local colour based on reflectivity
        colour = colour.mul(1-material.reflectivity).add(reflected_colour.mul(material.reflectivity))

    # Refraction handling
    if (material and material.isrefractive and (depth < scene.nbounces)):
        if (normal.dot(ray.dir) < 0):
            eta = 1 / material.refractiveindex
        else:
            eta = material.refractiveindex
        refraction_dir = refract(ray.dir, normal, eta)
        if (not refraction_dir.abs2()): # Valid refraction
            refracted_ray = Line(point, refraction_dir)
            # Trace the refracted ray
            refracted_colour = trace_ray(refracted_ray, scene, depth+1)
            # Blend the refracted colour with the final colour
            colour = colour.mul(1-material.reflectivity).add(refracted_colour.mul(material.reflectivity))
    return colour.clamp(0, 1)


def raytracer(scene, hasher):
    width = scene.camera.width
    height = scene.camera.height
    image = [scene.bg]*(width*height)
    # Precompute camera parameters
    aspect_ratio = width/height
    viewport_height = 2*math.tan(scene.camera.fov/2)
    viewport_width = aspect_ratio * viewport_height
    forward = scene.camera.looking_dir
    up = scene.camera.upwards_dir
    right = forward.cross(up).unit()
    root_pos = scene.camera.pos

    y = -1
    while True:
        y += 1
        if (y >= height):
            break
        x = -1
        while True:
            x += 1
            if (x >= width):
                break
            colour = Colour(Vector(0, 0, 0))
            total_pixel_samples = PIXEL_SAMPLES
            pixel_sample = -1
            while True:
                pixel_sample += 1
                if (pixel_sample >= total_pixel_samples):
                    break

                if (PIXEL_SAMPLES > 1):
                    sample_x = x + float_random()
                    sample_y = y + float_random()
                else:
                    sample_x = x
                    sample_y = y

                lens_sample = -1
                while True:
                    lens_sample += 1
                    if (lens_sample >= LENS_SAMPLES):
                        break

                    if (LENS_SAMPLES > 1):
                        pos = root_pos.add(up.mul(float_random()/100).add(right.mul(float_random()/100)))
                    else:
                        pos = root_pos
                    xmul = viewport_width*(sample_x/width - 0.5)
                    ymul = viewport_height*(sample_y/height - 0.5)
                    pixel_dir = forward.add(right.mul(-xmul)).add(up.mul(-ymul))

                    ray = Line(pos, pixel_dir.unit())
                    colour = colour.add(trace_ray(ray, scene, 0))
            image[y*width+x] = colour.div(PIXEL_SAMPLES*LENS_SAMPLES)
    tone_map(image, width, height, scene.camera)
    return hasher(image, width, height)


########################################################################### Main
def load_scene():
    WIDTH  = 600
    HEIGHT = 400
    # WIDTH  = 120
    # HEIGHT = 80
    FORWARDS = Vector(0.447249, -0.157593, 0.880416)
    UP       = Vector(0.0, 1.0, 0.0)
    POS      = Vector(-1, 0.5, -1.5)
    BG       = Colour(Vector(0.25, 0.25, 0.25))
    NBOUNCES = 8
    EXPOSURE = 0.1

    def make_material(a, b):
        return Material(0.1, 0.9, 20, Colour(a), Colour(Vector(1,1,1)), b, 1, False, 1)
    c1 = Vector(0.8,0.5,0.5)
    c2 = Vector(0.5,0.5,0.8)
    c3 = Vector(0.8,0.5,0.8)
    c4 = Vector(0.5,0.8,0.5)

    objs = []
    objs.append(Sphere(make_material(c1, False), Vector(-0.35,-0.2,1), 0.3))
    objs.append(Sphere(make_material(c3, False), Vector(0,0.2,-1.25), 0.2))
    objs.append(Cylinder(make_material(c2, False), Vector(0,1,0), Vector(0.3,0,1), 0.5, 0.25))
    objs.append(Triangle(make_material(c4, False), Vector(-1,-0.5,2), Vector(1,-0.5,2), Vector(1,-0.5,0)))
    objs.append(Triangle(make_material(c4, False), Vector(-1,-0.5,0), Vector(-1,-0.5,2), Vector(1,-0.5,0)))
    objs.append(Triangle(make_material(c4, True), Vector(-1,-0.5,2), Vector(1,2.5,2), Vector(1,-0.5,2)))
    objs.append(Triangle(make_material(c4, True), Vector(-1,-0.5,2), Vector(-1,2.5,2), Vector(1,2.5,2)))

    lights = []
    lights.append(PointLight(Colour(Vector(0.75,0.75,0.75)), Vector(0,1,0.5)))

    # FORWARDS = Vector(0, 0, 1)
    # POS      = Vector(0, 0, 0)
    # objs.append(Sphere(make_material(c1, False), Vector(0,0,1.5), 0.3))

    camera = Camera(WIDTH, HEIGHT, FORWARDS, UP, POS, 45, EXPOSURE)
    return Scene(lights, objs, camera, BG, NBOUNCES)


NO_INTERSECTION = Intersection(None, None)

class io:
    print = print


def hash_img(img, width, height):
    hash = 0
    for y in range(height):
        for x in range(width):
            colour = img[y*width+x]
            r, g, b = colour.r(), colour.g(), colour.b()
            colour_weight = 3*r + 5*g + 7*b
            x_weight = 1 - (x-width/2)**2 / width
            y_weight = 1 - (y-height/2)**2 / height
            hash += colour_weight * x_weight * y_weight
    return hash/(width*height)**2

def main():
    scene = load_scene()
    return raytracer(scene, hash_img)

PIXEL_SAMPLES = 1
LENS_SAMPLES = 1

print(f"{main():.8f}")