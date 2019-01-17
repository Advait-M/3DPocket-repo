from OpenGLContext import visitor
from numpy import *
from OpenGLContext.loaders import vrml97

from OpenGLContext.loaders.loader import Loader
import cPickle as pickle


def findShapes(sg):
    """ Find all shape nodes, return the list of shape nodes
    and commulative transforms associated with these nodes.
    """

    paths_to_shapes = visitor.find(sg, (vrml97.basenamespaces.basenodes.Shape,))
    # last elements of all paths are
    shapes = map(lambda x: x[-1], paths_to_shapes)
    transforms = map(lambda x: x.transformMatrix(), paths_to_shapes)
    return shapes, transforms

def extractPolys(shapes, transforms):
    """ Finds all indexed face sets in the list of shapes
    returns an Nx3 array of vertex coordinates and
    a list of integer arrays for faces
    extracting color information can be added here.
    """
    vertices = array([[0, 0, 0]])
    faces = []
    vertind_offset = 1
    for (s, T) in zip(shapes, transforms):
        if isinstance(s.geometry, vrml97.basenamespaces.basenodes.IndexedFaceSet):
            g = s.geometry
            transf_vert = g.coord.point

            # add vertices to the global list, increment the offset in the
            # global vertex list by the # of vertices added
            vertices = concatenate((vertices, transf_vert))
            fv_inds = arange(0, len(g.coordIndex))
            # array of indices of -1's in coordIndex
            face_terminators = fv_inds[g.coordIndex == -1]
            # split the array after each -1
            shape_faces = split(g.coordIndex, face_terminators + 1)
            # if there is a terminating -1 some faces may be 0 length
            shape_faces = [x for x in shape_faces if len(x) > 0]

            # chop off -1's and add faces to the global list
            # add offset to all vertex indices
            faces = faces + map(lambda x: x[0:-1] + vertind_offset, shape_faces)
            vertind_offset += len(g.coord.point)
    return vertices, faces

def saveSurfacePickle(fileName):
    """Extracts triangular faces from WRL file and saves them as a pickle."""
    in_sg = Loader.load(fileName + ".wrl")
    shapes, transforms = findShapes(in_sg)
    V, F = extractPolys(shapes, transforms)
    print("Done vertices and faces extraction from WRL file (Connolly surface).")
    surfaceTrianglesWRL = []
    for i in range(0, len(F)):
        temp = []
        for j in F[i]:
            temp.append(tuple(V[j]))
        surfaceTrianglesWRL.append(temp)

    # Save pickle with data
    with open(fileName + ".pkl", "w") as f:
        pickle.dump(surfaceTrianglesWRL, f)
    print("Saved Connolly surface WRL triangular faces pickle.")