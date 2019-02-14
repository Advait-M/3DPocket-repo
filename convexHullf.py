from mathutils import *
import operator
import random
import itertools
import numpy as np
import math
from pymol import cmd, stored
import cPickle as pickle
import wrlparse
from scipy.optimize import linprog
import pymol
from pymol.cgo import *
import time
from Bio.PDB import *

# Based on algorithm outlined in the quickhull Python module
def qdome2d(vertices, base, normal, precision=0.0001):
    """
    Builds a convex hull on top of two base vertices with specified normal.
    Note that this is a helper function for qhull2d.
    Returns a list of vertices that make up a fan of the dome.
    """

    vert0, vert1 = base
    outer = [(dist, vert)
             for dist, vert
             in zip((vecDotProduct(vecCrossProduct(normal,
                                                   vecSub(vert1, vert0)),
                                   vecSub(vert, vert0))
                     for vert in vertices),
                    vertices)
             if dist > precision]

    if outer:
        pivot = max(outer, key=lambda i: i[0])[1]
        outer_verts = map(operator.itemgetter(1), outer)
        return qdome2d(outer_verts, [vert0, pivot], normal, precision) \
               + qdome2d(outer_verts, [pivot, vert1], normal, precision)[1:]
    else:
        return base


def qhull2d(vertices, normal, precision=0.0001):
    """
    Implements the 2D quickhull algorithm in 3D for vertices viewed in the direction of the normal.
    Returns a fan of vertices that make up this surface (list of extreme points).
    """
    base = basesimplex3d(vertices, precision)
    if len(base) >= 2:
        vert0, vert1 = base[:2]
        return qdome2d(vertices, [vert0, vert1], normal, precision) \
               + qdome2d(vertices, [vert1, vert0], normal, precision)[1:-1]
    else:
        return base


def basesimplex3d(vertices, precision=0.0001):
    """
    Finds the four extreme points, which are to be used as 
    a starting base for the quick hull algorithm.
    Ideally, these four points should be as far apart as possible to speed up
    the quickhull algorithm.
    """
    # sort axes by their extent in vertices
    extents = sorted(range(3),
                     key=lambda i:
                     max(vert[i] for vert in vertices)
                     - min(vert[i] for vert in vertices))
    # extents[0] has the index with largest extent etc.
    # so let us minimize and maximize vertices with key
    # (vert[extents[0]], vert[extents[1]], vert[extents[2]])
    # which we can write as operator.itemgetter(*extents)(vert)
    vert0 = min(vertices, key=operator.itemgetter(*extents))
    vert1 = max(vertices, key=operator.itemgetter(*extents))
    # check if all vertices coincide
    if vecDistance(vert0, vert1) < precision:
        return [vert0]
    # as a third extreme point select that one which maximizes the distance
    # from the vert0 - vert1 axis
    vert2 = max(vertices,
                key=lambda vert: vecDistanceAxis((vert0, vert1), vert))
    # check if all vertices are colinear
    if vecDistanceAxis((vert0, vert1), vert2) < precision:
        return [vert0, vert1]
    # as a fourth extreme point select one which maximizes the distance from
    # the v0, v1, v2 triangle
    vert3 = max(vertices,
                key=lambda vert: abs(vecDistanceTriangle((vert0, vert1, vert2),
                                                         vert)))
    # ensure positive orientation and check if all vertices are coplanar
    orientation = vecDistanceTriangle((vert0, vert1, vert2), vert3)
    if orientation > precision:
        return [vert0, vert1, vert2, vert3]
    elif orientation < -precision:
        return [vert1, vert0, vert2, vert3]
    else:
        # coplanar
        return [vert0, vert1, vert2]


def qhull3d(vertices, precision=0.0001, verbose=False):
    """
    Returns the triangles that make up the convex hull of the vertices.
    Distances less than the specified precision are considered to be 0 to simplify
    hulls of complex meshes. Returns a list containing the extreme points and a list
    containing the triangular faces of the convex hull.
    """
    # find a simplex to start from
    hull_vertices = basesimplex3d(vertices, precision)

    # handle degenerate cases
    if len(hull_vertices) == 3:
        # coplanar
        # hull_vertices = qhull2d(vertices, vecNormalized(vecNormal(*hull_vertices)),precision)
        hull_vertices = qhull2d(vertices, vecNormal(*hull_vertices), precision)
        # return hull_vertices, [ (0, i+1, i+2)
        #                        for i in xrange(len(hull_vertices) - 2) ]
        return hull_vertices, [(i, i + 1, len(hull_vertices) - 1)
                               for i in xrange(len(hull_vertices) - 2)]
    elif len(hull_vertices) <= 2:
        # colinear or singular
        # no triangles for these cases
        return hull_vertices, []
    # print '3d'
    # print vertices
    # raw_input()
    # construct list of triangles of this simplex
    hull_triangles = set([operator.itemgetter(i, j, k)(hull_vertices)
                          for i, j, k in ((1, 0, 2), (0, 1, 3), (0, 3, 2), (3, 1, 2))])

    if verbose:
        print("starting set", hull_vertices)

    # construct list of outer vertices for each triangle
    outer_vertices = {}
    for triangle in hull_triangles:
        outer = \
            [(dist, vert)
             for dist, vert
             in zip((vecDistanceTriangle(triangle, vert)
                     for vert in vertices),
                    vertices)
             if dist > precision]
        if outer:
            outer_vertices[triangle] = outer

    # as long as there are triangles with outer vertices
    while outer_vertices:
        # grab a triangle and its outer vertices
        tmp_iter = iter(outer_vertices.items())
        triangle, outer = next(tmp_iter)  # tmp_iter trick to make 2to3 work
        # calculate pivot point
        pivot = max(outer)[1]
        if verbose:
            print("pivot", pivot)
        # add it to the list of extreme vertices
        hull_vertices.append(pivot)
        # and update the list of triangles:
        # 1. calculate visibility of triangles to pivot point
        visibility = [vecDistanceTriangle(othertriangle, pivot) > precision
                      for othertriangle in outer_vertices.keys()]
        # 2. get list of visible triangles
        visible_triangles = [othertriangle
                             for othertriangle, visible
                             in zip(outer_vertices.keys(), visibility)
                             if visible]
        # 3. find all edges of visible triangles
        visible_edges = []
        for visible_triangle in visible_triangles:
            visible_edges += [operator.itemgetter(i, j)(visible_triangle)
                              for i, j in ((0, 1), (1, 2), (2, 0))]
        if verbose:
            print("visible edges", visible_edges)
        # 4. construct horizon: edges that are not shared with another triangle
        horizon_edges = [edge for edge in visible_edges
                         if not tuple(reversed(edge)) in visible_edges]
        # 5. remove visible triangles from list
        # this puts a hole inside the triangle list
        visible_outer = set()
        for outer_verts in outer_vertices.values():
            visible_outer |= set(map(operator.itemgetter(1), outer_verts))
        for triangle in visible_triangles:
            if verbose:
                print("removing", triangle)
            hull_triangles.remove(triangle)
            del outer_vertices[triangle]
        # 6. close triangle list by adding cone from horizon to pivot
        # also update the outer triangle list as we go
        for edge in horizon_edges:
            newtriangle = edge + (pivot,)
            newouter = \
                [(dist, vert)
                 for dist, vert in zip((vecDistanceTriangle(newtriangle,
                                                            vert)
                                        for vert in visible_outer),
                                       visible_outer)
                 if dist > precision]
            hull_triangles.add(newtriangle)
            if newouter:
                outer_vertices[newtriangle] = newouter
            if verbose:
                print("adding", newtriangle, newouter)

    # no triangle has outer vertices anymore
    # so the convex hull is complete!
    return hull_vertices, hull_triangles

def addVectors(a, b):
    """Add 3D vectors a and b."""
    return((a[0]+b[0], a[1]+b[1], a[2]+b[2]))

def subtractVectors(a, b):
    """Subtract 3D vector b from 3D vector a."""
    return((a[0]-b[0], a[1]-b[1], a[2]-b[2]))

def findCentroid(triangle):
    """Returns centroid of given triangle."""
    return (((triangle[0][0] + triangle[1][0] + triangle[2][0])/3), ((triangle[0][1] + triangle[1][1] + triangle[2][1])/3), ((triangle[0][2] + triangle[1][2] + triangle[2][2])/3))

def cross(a, b):
    """Returns the cross product of vectors a and b."""
    c = (a[1]*b[2] - a[2]*b[1],
         a[2]*b[0] - a[0]*b[2],
         a[0]*b[1] - a[1]*b[0])
    return c

def getScalarEqnCoeffs(triangle):
    """Get scalar equation coefficients of the plane defined by the 3 vertices of the given triangle."""
    a = triangle[0]
    b = triangle[1]
    c = triangle[2]
    ab = subtractVectors(b,a)
    ac = subtractVectors(c,a)
    res = cross(ab, ac)
    A = res[0]
    B = res[1]
    C = res[2]
    D = -A*a[0] - B*a[1] - C*a[2]
    return (A, B, C, D)

def getDistancePlanePoint(coeffs, point):
    """Calculate distance from point to plane defined by the coefficients within its scalar equation."""
    top = abs(coeffs[0]*point[0] + coeffs[1]*point[1] + coeffs[2]*point[2] + coeffs[3])
    bottom = (coeffs[0]**2 + coeffs[1]**2 + coeffs[2]**2)**0.5
    return top/bottom

def getColours(scores):
    """Normalize distance scores to get colour values (0-1)."""
    maxi = max(scores)
    mini = min(scores)
    rang = maxi-mini

    values = []
    for i in scores:
        values.append(((i-mini)/rang, (i-mini)/rang, (i-mini)/rang))
    return values

def checkInSphere(coord, center, r):
    """Check if coordinate is within sphere."""
    return (coord[0] - center[0]) ** 2 + (coord[1] - center[1]) ** 2 + (coord[2] - center[2]) ** 2 <= r ** 2

def getResults(proteinName):
    """Run 3DPocket on desired protein. Protein file as CIF must be within root directory."""
    getStats = False
    # Used to parse CIF files
    parser = MMCIFParser()
    structure = parser.get_structure(proteinName, proteinName + ".cif")

    # Store all atom coordinates
    atoms = []
    for atom in structure.get_atoms():
        atoms.append(tuple(atom.get_coord()))
    print("Amount of atoms:", len(atoms))

    # Maps residues to their corresponding atoms
    residueAtomDict = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                residueAtomDict[residue.id[1]] = [tuple(x.get_coord()) for x in residue.get_atoms()]
    print("Amount of residues:", len(residueAtomDict.keys()))

    # Find the vertices and triangular faces of the 3D convex hull
    hullVertices, hullTriangles = qhull3d(atoms)

    print("Amount of triangular faces in convex hull:", len(hullTriangles))

    # Finish launching PyMol
    pymol.finish_launching()
    # Load desired protein
    pymol.cmd.load(proteinName + ".cif", proteinName)
    # Set background colour to white
    pymol.cmd.bg_color("white")

    # CGO requirements (must start with these constants for PyMol CGO)
    convexHullTrianglesCGO = [BEGIN, TRIANGLES]

    # Create CGO representation of convex hull
    for i in hullTriangles:
        convexHullTrianglesCGO.append(COLOR)
        for colours in range(0, 3):
            convexHullTrianglesCGO.append(random.random())
        for j in i:
            convexHullTrianglesCGO.append(VERTEX)
            for k in j:
                convexHullTrianglesCGO.append(k)

    # Load convex hull CGO into PyMol
    pymol.cmd.load_cgo(convexHullTrianglesCGO, "convexHull_" + proteinName)

    # Hide all objects
    pymol.cmd.hide('everything')
    # Compute the Connolly surface of the protein
    pymol.cmd.show("surface", proteinName)
    # Reset view so the PyMol WRL coordinates match the PDB/CIF file
    pymol.cmd.reset()
    pymol.cmd.origin(position=[0, 0, 0])
    pymol.cmd.center("origin")
    pymol.cmd.move('z', -cmd.get_view()[11])
    # Save Connolly surface to WRL file
    pymol.cmd.save(proteinName + ".wrl")

    # Parse WRL file into Python objects and save to a Pickle file
    wrlparse.saveSurfacePickle(proteinName)

    # Open Pickle file
    with open(proteinName + ".pkl", "r") as f:
        surfaceTriangles = pickle.load(f)

    # Display the Connolly surface with each triangular face having a randomized colour
    randomSurfaceTrianglesCGO = [BEGIN, TRIANGLES]
    counter = 0
    for i in surfaceTriangles:
        randomSurfaceTrianglesCGO.append(COLOR)
        # Assign random colours to each triangular face
        for colours in range(0, 3):
            randomSurfaceTrianglesCGO.append(random.random())
        for j in i:
            randomSurfaceTrianglesCGO.append(VERTEX)
            for k in j:
                randomSurfaceTrianglesCGO.append(k)
        counter += 1

    print("Amount of triangular faces in Connolly surface:", len(randomSurfaceTrianglesCGO))
    pymol.cmd.load_cgo(randomSurfaceTrianglesCGO, "randomColouredSurface_" + proteinName)
    pymol.cmd.show("cgo")

    # Initialize scores list (stores minimum distances) with positive infinity values
    scores = [float("inf")] * len(surfaceTriangles)

    # Stores all coefficients of the scalar equations representing the planes defined by triangular faces within the convex hull
    allCoeffs = []
    print("Finding coefficients of scalar equations.")
    for j in hullTriangles:
        coeffsj = getScalarEqnCoeffs(j)
        allCoeffs.append(coeffsj)

    print("Finding centroids and minimum distances.")
    centroids = []
    for i in range(0, len(surfaceTriangles)):
        centroids.append(findCentroid(surfaceTriangles[i]))
        for j in range(0, len(hullTriangles)):
            coeffscur = allCoeffs[j]
            score = getDistancePlanePoint(coeffscur, centroids[i])
            if score < scores[i]:
                scores[i] = score

    print("Getting colour values.")
    colourVals = getColours(scores)

    # Display raw colourized protein (white indicates likely binding site and black indicates nonbinding sites)
    print("Starting raw colourized protein computation.")
    rawTrianglesCGO = [BEGIN, TRIANGLES]
    counter = 0
    for i in surfaceTriangles:
        rawTrianglesCGO.append(COLOR)
        for colours in range(0, 3):
            rawTrianglesCGO.append(colourVals[counter][colours])
        for j in i:
            rawTrianglesCGO.append(VERTEX)
            for k in j:
                rawTrianglesCGO.append(k)
        counter += 1
    pymol.cmd.load_cgo(rawTrianglesCGO, "rawColourizedSurface_" + proteinName)
    pymol.cmd.show("cgo")

    print("Starting predicted binding site (threshold structure) computation.")
    # Display predicted binding sites (shown in red)
    thresholdTrianglesCGO = [BEGIN, TRIANGLES]
    counter = 0
    resc = 0
    for i in surfaceTriangles:
        thresholdTrianglesCGO.append(COLOR)
        # Define a threshold to differentiate binding sites vs nonbinding sites (colour values are normalized from 0-1)
        if colourVals[counter][0] > 0.4:
            thresholdTrianglesCGO.append(0.7)
            thresholdTrianglesCGO.append(0)
            thresholdTrianglesCGO.append(0)
            # Map predicted binding site triangular faces within the Connolly surface to predicted binding sites
            # Allows for comparison to LIGASITE actual binding sites and other algorithms
            # Any atom within 1.7 A of a triangle centroid is considered to be a binding site atom and the residue containing that atom is considered a binding site residue
            if getStats:
                for j in residueAtomDict.keys():
                    if residueAtomDict[j] != True:
                        for k in residueAtomDict[j]:
                            if checkInSphere(k, centroids[counter], 1.7):
                                residueAtomDict[j] = True
                                resc += 1
                                break
        else:
            # Colour non-binding sites grey
            thresholdTrianglesCGO.append(0.2)
            thresholdTrianglesCGO.append(0.2)
            thresholdTrianglesCGO.append(0.2)
        for j in i:
            thresholdTrianglesCGO.append(VERTEX)
            for k in j:
                thresholdTrianglesCGO.append(k)
        counter += 1

    print("Amount of triangular faces within the threshold structure:", len(thresholdTrianglesCGO))
    pymol.cmd.load_cgo(thresholdTrianglesCGO, "thresholdSurface_" + proteinName)

    # Get predicted binding residues
    if getStats:
        finals = []
        for j in residueAtomDict.keys():
            if residueAtomDict[j] == True:
                finals.append(j)
        print("Binding residues:", finals)

    # Allow for different thresholds to be tested without restarting the whole process
    while True:
        ret, count2 = changeThreshold(float(input("Enter threshold: ")), surfaceTriangles, colourVals)
        pymol.cmd.load_cgo(ret, str(count2) + "thresholdSurface_" + proteinName)
        pymol.cmd.show("cgo")

def changeThreshold(thresholdValue, surfaceTriangles, colourVals):
    """Add new structure with the desired threshold value."""
    print("Changing threshold to:", thresholdValue)
    newThresholdTrianglesCGO = [BEGIN, TRIANGLES]
    counter = 0
    for i in surfaceTriangles:
        newThresholdTrianglesCGO.append(COLOR)
        if colourVals[counter][0] > thresholdValue:
            newThresholdTrianglesCGO.append(0.7)
            newThresholdTrianglesCGO.append(0)
            newThresholdTrianglesCGO.append(0)
        else:
            newThresholdTrianglesCGO.append(0.2)
            newThresholdTrianglesCGO.append(0.2)
            newThresholdTrianglesCGO.append(0.2)
        for j in i:
            newThresholdTrianglesCGO.append(VERTEX)
            for k in j:
                newThresholdTrianglesCGO.append(k)
        counter += 1
    return newThresholdTrianglesCGO, thresholdValue

proteins = ["2cwh"]#"1l1o", "1rev", "3lpo", "3ck4", "3gzk"] "1g6c" "2cwh" "3cwk"
for i in proteins:
    getResults(i)
