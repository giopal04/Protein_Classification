import numpy as np
import os
#from symmetries import SymmetryPlane, SymmetryAxis
from symmetria.transformations import *

#Stores the information of a single benchmark shape. The benchmark shape consists of a point cloud and a set of symmetries
class BenchmarkShape:
    def __init__(self, points, symmetry_list = None):
        self.points = points #Geometry (point cloud)
        self.symmetry_list = symmetry_list #List of SymmetryPlane
    
    #Adds a symmetry to the set
    def add_symmetry(self, symmetry):
        self.symmetry_list.append(symmetry)
    
    #Applies a rotation to the benchmark shape. It rotates the shape and applies the transformation to each individual plane
    def apply_rotation(self, rot):
        ones = np.ones((1,self.points.shape[0]))
        self.coords = np.concatenate((self.points.T, ones))
        
        self.coords = rot@self.coords
        
        self.points = self.coords[0:3,:].T

        for sym in self.symmetry_list:
            sym.apply_rotation(rot)
    
    def apply_traslation(self, x, y, z):
        #print(self.points.shape)
        self.points[:,0] = self.points[:,0] + x
        self.points[:,1] = self.points[:,1] + y
        self.points[:,2] = self.points[:,2] + z

        #print('New shape')
        for sym in self.symmetry_list:
            #print(type(sym))
            #print('Before:', sym.point)
            sym.apply_traslation(x, y, z)
            #print('After:', sym.point)

    
    #Applies uniform noise. n is a parameter for the amount of noise. T is the number of points to apply
    def apply_uniform_noise(self, n, T):
        num_points = self.points.shape[0]

        indices = np.random.choice(num_points, T, replace=False)

        self.points[indices, :] = self.points[indices,:] + (2*np.random.rand(T, 3)-1)/n
    
    def apply_gaussian_noise(self, n, T):
        num_points = self.points.shape[0]

        indices = np.random.choice(num_points, T, replace=False)

        self.points[indices, :] = self.points[indices,:] + np.random.randn(T, 3)/n
    
    #Applies undersampling in the point cloud. T points are removed
    def apply_undersampling(self, T):
        num_points = self.points.shape[0]

        indicesRemoved = set(np.random.choice(num_points, T, replace=False))
        indicesTotal = set(range(num_points))

        indicesKept = list(indicesTotal - indicesRemoved)

        self.points = self.points[indicesKept,:] 
    
    #Saves the point cloud and the symmetries
    def save_benchmark_shape(self, output_path, prefix, file_fmt='npz', number_fmt='%.18e'):
        #pc_path = os.path.join(output_path, prefix + '.txt')
        #np.savetxt(pc_path, self.points, fmt='%3.6f')
        pc_path = os.path.join(output_path, prefix[:-1] + '.' + file_fmt)
        if file_fmt == 'npz':
            np.savez_compressed(pc_path, points=self.points, fmt=number_fmt)
        elif file_fmt == 'txt':
            np.savetxt(pc_path, self.points, fmt=number_fmt)
        elif file_fmt == 'gz':
            import gzip
            f = gzip.GzipFile(f"{pc_path}", "w")
            #np.save(file=f, arr=self.points)
            np.savetxt(f, self.points, fmt=number_fmt)
            f.close()
        elif file_fmt == 'xz':
            import lzma
            f = lzma.open(f"{pc_path}", 'wb')
            #np.save(file=f, arr=self.points)
            np.savetxt(f, self.points, fmt=number_fmt)
            f.close()
        
        sym_path = os.path.join(output_path, prefix + 'sym.txt')
        nfmt = number_fmt.replace('%','')

        if not self.symmetry_list or len(self.symmetry_list) == 0:
            return

        with open(sym_path, 'wt') as f:
            f.write(str(len(self.symmetry_list))+'\n')

            for sym in self.symmetry_list:
                if isinstance(sym, SymmetryPlane):
                    f.write("plane ")
                    f.write(" ".join([str(f'{x:{nfmt}}') for x in list(sym.normal)]) + ' ')
                    f.write(" ".join([str(f'{x:{nfmt}}') for x in list(sym.point)]) + '\n')
                elif isinstance(sym, SymmetryAxis):
                    f.write("axis ")
                    f.write(" ".join([str(f'{x:{nfmt}}') for x in list(sym.normal)]) + ' ')
                    f.write(" ".join([str(f'{x:{nfmt}}') for x in list(sym.point)]) + ' ')
                    f.write(str(f'{sym.angle:{nfmt}}') + '\n')

                #f.write(" ".join([str(x) for x in list(sym.normal)]) + ' ')
                #f.write(" ".join([str(x) for x in list(sym.point)]) + '\n')


#Given a 2D curve, this method extrudes the curve 
def extrude_curve(X, Y, conic, rescale_z = True, minscale=0.5, maxscale=4.0):
    maxx = np.max(X)
    maxy = np.max(Y)

    maxtot = np.max([maxx, maxy])
    X = X / maxtot
    Y = Y / maxtot

    size = X.size
    
    scale = np.random.random(size)

    factor = (maxscale-minscale)*np.random.random() + minscale
    
    #Hardcoded factor
    #factor = 3.721247281765034
    
    Z = factor*scale - factor/2
    #print(f'Max Z: {np.max(Z)}')
    
    if conic:
        points = np.concatenate(((scale*X).reshape(size, 1),(scale*Y).reshape(size,1), Z.reshape(size,1)), axis=1)
    else:
        points = np.concatenate((X.reshape(size, 1),Y.reshape(size,1), Z.reshape(size,1)), axis=1)
    
    return points

def generateT(t):
    return np.array([[1, t, t**2, t**3]]).T

def bezierMatrix(P0, P1, P2, P3):
    
    # Generate a matrix concatenating the columns
    G = np.concatenate((P0, P1, P2, P3), axis=1)

    # Bezier base matrix is a constant
    Mb = np.array([[1, -3, 3, -1], [0, 3, -6, 3], [0, 0, 3, -3], [0, 0, 0, 1]])
    
    return np.matmul(G, Mb)

def evalCurve(M, N):
    # The parameter t should move between 0 and 1
    ts = np.linspace(0.0, 1.0, N)
    noise = (2*np.random.rand(N)-1)/(2*N)
    #noise = np.zeros(N)
    noise[0] = 0
    noise[-1] = 0

    ts = ts + noise
    
    # The computed value in R3 for each sample will be stored here
    curve = np.ndarray(shape=(N, 3), dtype=float)
    
    for i in range(len(ts)):
        T = generateT(ts[i])
        curve[i, 0:3] = np.matmul(M, T).T
        
    return curve

def rotatePoint(point, angle):
    R = rotationY(angle)
    P = np.expand_dims(point, axis=1)
    P = np.concatenate((P, np.ones((1,1))))
    P = matmul([R, P])[0:3,:].T
    return P

def revolution_surface(N = 80):
    x1 = 0.1*np.random.randint(1, 10)
    x2 = 0.1*np.random.randint(1, 10)
    x3 = 0.1*np.random.randint(1, 10)
    y1 = 0.1*np.random.randint(1, 10)
    y2 = 0.1*np.random.randint(0, 10)


    R0 = np.array([[x1, 1.0, 0.0]]).T #Coordenada Y no se mueve
    R1 = np.array([[x2, y1, 0.0]]).T
    R2 = np.array([[x3, y2, 0.0]]).T 
    R3 = np.array([[0.0, 0.0, 0.0]]).T #Este no se mueve

    #Hardcoded for Neurips figure
    R0 = np.array([[0.1, 1.0, 0.0]]).T #Coordenada Y no se mueve
    R1 = np.array([[0.3, 0.5, 0.0]]).T
    R2 = np.array([[0.8, 0.6, 0.0]]).T

    GMb = bezierMatrix(R0, R1, R2, R3)
    bezierCurve = evalCurve(GMb, N)
    L = [bezierCurve]
        
    angles = np.linspace(0.0, 2*np.pi, 100)
    #print(bezierCurve.shape)

    #for i in range(bezierCurve.shape[0]):
    #    for angle in angles:
    #        L.append(rotatePoint(bezierCurve[i,:], angle + 0.06*(2*np.random.rand()-1)))
    for angle in angles:
        bezierCurve=evalCurve(GMb, N)
        for i in range(bezierCurve.shape[0]):
            bezierCurve[i,:] = rotatePoint(bezierCurve[i,:], angle + 0.06*(2*np.random.rand()-1))
        L.append(bezierCurve)
        #R = rotationY(angle)
        #P = np.concatenate((bezierCurve.T, np.ones((1, bezierCurve.shape[0]))))
        #curve = matmul([R, P])[0:3,:].T
        #L.append(curve)
        #L.append(rotatePoint(bezierCurve, angle + 0.06*(2*np.random.rand()-1)))
        
    
    #for a in angles:
    #    R = rotationY(a)
    #    Z = np.concatenate((np.zeros((bezierCurve.shape[0], 1)),np.zeros((bezierCurve.shape[0], 1)),0.1*(2*np.random.rand(bezierCurve.shape[0], 1)-1)), axis=1)
    #    Z = bezierCurve + Z
    #    P = np.concatenate((Z.T, np.ones((1, bezierCurve.shape[0]))))
    #    curve = matmul([R, P])[0:3,:].T
    #    L.append(curve)
    
    curves = np.vstack(L)

    shape = BenchmarkShape(curves, [])
    shape.add_symmetry(SymmetryAxis(point=np.array([0.0,0.0,0.0]), normal=np.array([0.0,1.0,0.0]), angle=np.inf))

    return shape


#The following methods generate the shapes of our benchmark
def cylinder(a = 1.0, b=1.0, N = 80, conic=False):
    u = 2*np.pi*np.random.random(N**2) - np.pi
    
    X = a*np.cos(u)
    Y = b*np.sin(u)

    points = extrude_curve(X, Y, conic, rescale_z=False, minscale=0.5, maxscale=25.0)

    shape = BenchmarkShape(points,[])
    
    if not conic:
        shape.add_symmetry(SymmetryPlane(point=np.array([0.0,0.0,0.0]), normal=np.array([0.0,0.0,1.0])))
        shape.add_symmetry(SymmetryAxis(point=np.array([0.0,0.0,0.0]), normal=np.array([0.0,0.0,1.0]), angle=np.inf))
    else:
        shape.add_symmetry(SymmetryAxis(point=np.array([0.0,0.0,0.0]), normal=np.array([0.0,0.0,1.0]), angle=np.pi/2))

    return shape

def square(a=1.0, b=1.0, N=80, conic=False):
    u = 2*np.pi*np.random.random(N**2) - np.pi
    

    X = np.cos(u)
    Y = np.sin(u)

    C = np.vstack([X, Y])
    M = np.max(np.abs(C),axis=0)

    X1 = a*X/M
    Y1 = b*Y/M

    points = extrude_curve(X1, Y1, conic, rescale_z=False, minscale=0.5, maxscale=25.0)

    shape = BenchmarkShape(points,[])
    shape.add_symmetry(SymmetryPlane(point=np.array([0.0,0.0,0.0]), normal=np.array([1.0,0.0,0.0])))
    shape.add_symmetry(SymmetryPlane(point=np.array([0.0,0.0,0.0]), normal=np.array([0.0,1.0,0.0])))
    shape.add_symmetry(SymmetryPlane(point=np.array([0.0,0.0,0.0]), normal=np.array([1.0,1.0,0.0])))
    shape.add_symmetry(SymmetryPlane(point=np.array([0.0,0.0,0.0]), normal=np.array([-1.0,1.0,0.0])))
    
    shape.add_symmetry(SymmetryAxis(point=np.array([0.0,0.0,0.0]), normal=np.array([0.0,0.0,1.0]), angle=np.pi/2))

    if not conic:
        shape.add_symmetry(SymmetryPlane(point=np.array([0.0,0.0,0.0]), normal=np.array([0.0,0.0,1.0])))


    return shape

def citrus(a=1, b=1, N=80, conic = False):
    u = np.random.random(N**2)

    x = u - a/2
    y1 = np.sqrt(((a - u)**3*u**3)/a**4*b**2)
    y2 = -np.sqrt(((a - u)**3*u**3)/a**4*b**2)
    
    X = np.concatenate((x, x))
    Y = np.concatenate((y1,y2))

    points = extrude_curve(X, Y, conic)
    
    shape = BenchmarkShape(points,[])
    shape.add_symmetry(SymmetryPlane(point=np.array([0.0,0.0,0.0]), normal=np.array([1.0,0.0,0.0])))
    shape.add_symmetry(SymmetryPlane(point=np.array([0.0,0.0,0.0]), normal=np.array([0.0,1.0,0.0])))
    shape.add_symmetry(SymmetryAxis(point=np.array([0.0,0.0,0.0]), normal=np.array([0.0,0.0,1.0]), angle=np.pi))
    
    if not conic:
        shape.add_symmetry(SymmetryPlane(point=np.array([0.0,0.0,0.0]), normal=np.array([0.0,0.0,1.0])))
        shape.add_symmetry(SymmetryAxis(point=np.array([0.0,0.0,0.0]), normal=np.array([1.0,0.0,0.0]), angle=np.pi))
        shape.add_symmetry(SymmetryAxis(point=np.array([0.0,0.0,0.0]), normal=np.array([0.0,1.0,0.0]), angle=np.pi))
    
    return shape

#Test function only to generate a bad extrusion (figure paper Neurips)
def m_convexities_bad(a=0.1, b=0.4, num = 4, N=80, conic=False):
    u = 2*np.pi*np.random.random(N**2) - np.pi
    
    X = (a*np.cos(u))/(1+b*np.cos(num*u))
    Y = (a*np.sin(u))/(1+b*np.cos(num*u))

    factor = 3.721247281765034
    L = []
    for i in range(101):
        Z = np.ones(N**2)*((i/100)*factor - factor/2)
        points = np.concatenate((X.reshape(N**2, 1),Y.reshape(N**2,1), Z.reshape(N**2,1)), axis=1)
        L.append(points)
    
    points = np.vstack(L)

    shape = BenchmarkShape(points,[])
    shape.add_symmetry(SymmetryAxis(point=np.array([0.0,0.0,0.0]), normal=np.array([0.0,0.0,1.0]), angle=(2*np.pi)/num))

    return shape



def m_convexities(a=0.1, b=0.4, num = 4, N=80, conic=False):

    u = 2*np.pi*np.random.random(N**2) - np.pi
    
    X = (a*np.cos(u))/(1+b*np.cos(num*u))
    Y = (a*np.sin(u))/(1+b*np.cos(num*u))

    points = extrude_curve(X, Y, conic)

    shape = BenchmarkShape(points,[])
    shape.add_symmetry(SymmetryAxis(point=np.array([0.0,0.0,0.0]), normal=np.array([0.0,0.0,1.0]), angle=(2*np.pi)/num))

    for angle in np.arange(0, np.pi, np.pi/num):
        shape.add_symmetry(SymmetryPlane(point=np.array([0.0,0.0,0.0]), normal=np.array([np.sin(angle),np.cos(angle),0.0])))

    if not conic:
        shape.add_symmetry(SymmetryPlane(point=np.array([0.0,0.0,0.0]), normal=np.array([0.0,0.0,1.0])))
        shape.add_symmetry(SymmetryAxis(point=np.array([0.0,0.0,0.0]), normal=np.array([1.0,0.0,0.0]), angle=np.pi))
        if num % 2 == 0:
            shape.add_symmetry(SymmetryAxis(point=np.array([0.0,0.0,0.0]), normal=np.array([0.0,1.0,0.0]), angle=np.pi))

    return shape

def geometric_petal_b(a=2, b=1, num=3, N=80, conic=False):
    u = 2*np.pi*np.random.random(N**2) - np.pi

    X = (a + b*(np.cos(2*num*u)))*np.cos(u)
    Y = (a + b*(np.cos(2*num*u)))*np.sin(u)

    points = extrude_curve(X, Y, conic)

    shape = BenchmarkShape(points,[])
    
    shape.add_symmetry(SymmetryAxis(point=np.array([0.0,0.0,0.0]), normal=np.array([0.0,0.0,1.0]), angle=np.pi/num))
    
    for angle in np.arange(0, np.pi, np.pi/(num*2)):
        shape.add_symmetry(SymmetryPlane(point=np.array([0.0,0.0,0.0]), normal=np.array([np.sin(angle),np.cos(angle),0.0])))
    
    if not conic:
        shape.add_symmetry(SymmetryPlane(point=np.array([0.0,0.0,0.0]), normal=np.array([0.0,0.0,1.0])))
        shape.add_symmetry(SymmetryAxis(point=np.array([0.0,0.0,0.0]), normal=np.array([1.0,0.0,0.0]), angle=np.pi))
        shape.add_symmetry(SymmetryAxis(point=np.array([0.0,0.0,0.0]), normal=np.array([0.0,1.0,0.0]), angle=np.pi))
        
    return shape

def lemniscate_bernoulli(a = 1, N = 80, conic=False):
    u = 2*np.pi*np.random.random(N**2) - np.pi

    X = a * (np.sin(u)/(1 + np.cos(u)**2))
    Y = a * (np.sin(u)*np.cos(u)/(1 + np.cos(u)**2))

    points = extrude_curve(X, Y, conic)
   
    shape = BenchmarkShape(points,[])
    shape.add_symmetry(SymmetryPlane(point=np.array([0.0,0.0,0.0]), normal=np.array([1.0,0.0,0.0])))
    shape.add_symmetry(SymmetryPlane(point=np.array([0.0,0.0,0.0]), normal=np.array([0.0,1.0,0.0])))
    shape.add_symmetry(SymmetryAxis(point=np.array([0.0,0.0,0.0]), normal=np.array([0.0,0.0,1.0]), angle=np.pi))

    if not conic:
        shape.add_symmetry(SymmetryPlane(point=np.array([0.0,0.0,0.0]), normal=np.array([0.0,0.0,1.0])))
        shape.add_symmetry(SymmetryAxis(point=np.array([0.0,0.0,0.0]), normal=np.array([1.0,0.0,0.0]), angle=np.pi))
        shape.add_symmetry(SymmetryAxis(point=np.array([0.0,0.0,0.0]), normal=np.array([0.0,1.0,0.0]), angle=np.pi))
    
    return shape

def egg_keplero(a = 1, N=80, conic=False):
    u = 8*np.random.random(N**2) - 4

    X = a/((1 + u**2)**2) - a/2
    Y = a*u/((1 + u**2)**2)

    points = extrude_curve(X, Y, conic)

    #points[:,0] = points[:,0] + a/2
    
    shape = BenchmarkShape(points,[])
    #shape.add_symmetry(SymmetryPlane(point=np.array([a/2,0.0,0.0]), normal=np.array([0.0,1.0,0.0])))
    shape.add_symmetry(SymmetryPlane(point=np.array([0.0,0.0,0.0]), normal=np.array([0.0,1.0,0.0])))

    if not conic:
        #shape.add_symmetry(SymmetryPlane(point=np.array([a/2,0.0,0.0]), normal=np.array([0.0,0.0,1.0])))
        shape.add_symmetry(SymmetryPlane(point=np.array([0.0,0.0,0.0]), normal=np.array([0.0,0.0,1.0])))
        shape.add_symmetry(SymmetryAxis(point=np.array([0.0,0.0,0.0]), normal=np.array([1.0,0.0,0.0]), angle=np.pi))

    return shape

def mouth_curve(a = 1, N=80, conic=False):
    u = 2*np.pi*np.random.random(N**2) - np.pi

    X = a*np.cos(u)
    Y = a*np.sin(u)**3

    points = extrude_curve(X, Y, conic)

    shape = BenchmarkShape(points,[])
    shape.add_symmetry(SymmetryPlane(point=np.array([0.0,0.0,0.0]), normal=np.array([1.0,0.0,0.0])))
    shape.add_symmetry(SymmetryPlane(point=np.array([0.0,0.0,0.0]), normal=np.array([0.0,1.0,0.0])))
    shape.add_symmetry(SymmetryAxis(point=np.array([0.0,0.0,0.0]), normal=np.array([0.0,0.0,1.0]), angle=np.pi))

    if not conic:
        shape.add_symmetry(SymmetryPlane(point=np.array([0.0,0.0,0.0]), normal=np.array([0.0,0.0,1.0])))
        shape.add_symmetry(SymmetryAxis(point=np.array([0.0,0.0,0.0]), normal=np.array([1.0,0.0,0.0]), angle=np.pi))
        shape.add_symmetry(SymmetryAxis(point=np.array([0.0,0.0,0.0]), normal=np.array([0.0,1.0,0.0]), angle=np.pi))
    
    
    return shape

def astroid(a = 1, N = 80, conic=False):
    u = 2*np.pi*np.random.random(N**2) - np.pi
    
    X = a*np.cos(u)**3
    Y = a*np.sin(u)**3

    points = extrude_curve(X, Y, conic)

    shape = BenchmarkShape(points,[])
    shape.add_symmetry(SymmetryPlane(point=np.array([0.0,0.0,0.0]), normal=np.array([0.0,1.0,0.0])))
    shape.add_symmetry(SymmetryPlane(point=np.array([0.0,0.0,0.0]), normal=np.array([1.0,0.0,0.0])))
    shape.add_symmetry(SymmetryPlane(point=np.array([0.0,0.0,0.0]), normal=np.array([1.0,1.0,0.0])))
    shape.add_symmetry(SymmetryPlane(point=np.array([0.0,0.0,0.0]), normal=np.array([-1.0,1.0,0.0])))
    shape.add_symmetry(SymmetryAxis(point=np.array([0.0,0.0,0.0]), normal=np.array([0.0,0.0,1.0]), angle=np.pi/2))

    if not conic:
        shape.add_symmetry(SymmetryPlane(point=np.array([0.0,0.0,0.0]), normal=np.array([0.0,0.0,1.0])))
        shape.add_symmetry(SymmetryAxis(point=np.array([0.0,0.0,0.0]), normal=np.array([1.0,0.0,0.0]), angle=np.pi))
        shape.add_symmetry(SymmetryAxis(point=np.array([0.0,0.0,0.0]), normal=np.array([0.0,1.0,0.0]), angle=np.pi))
        
    
    return shape

'''def cylinder2(N = 80):
    rMin = 1
    rMax = 6

    R = np.linspace(rMin, rMax, num=300)
    R1 = R[np.random.randint(0, R.size)]
    R2 = R[np.random.randint(0, R.size)]

    while np.abs(R2 - R1) < 0.5:
        R2 = R[np.random.randint(0, R.size)]
    
    #print(f'R1:{R1}')
    #print(f'R2:{R2}')

    alpha = np.pi*2

    hMin = 1
    hMax = 10
    h = np.linspace(hMin, hMax, 400)
    h = h[np.random.randint(0, h.size)]

    #print(f'h:{h}')

    u = np.random.random(N**2) * alpha
    v = np.random.random(N**2) * h

    X = R1 * np.cos(u)
    Y = R2 * np.sin(u)
    Z = v

    points = np.concatenate((X.reshape(N**2, 1),Y.reshape(N**2,1), Z.reshape(N**2,1)), axis=1)
    shape = BenchmarkShape(points,[])

    shape.add_symmetry(SymmetryPlane(point=np.array([0.0,0.0,0.0]), normal=np.array([0.0,0.0,1.0])))
    shape.add_symmetry(SymmetryAxis(point=np.array([0.0,0.0,0.0]), normal=np.array([0.0,0.0,1.0]), angle=np.inf))


    return shape'''
