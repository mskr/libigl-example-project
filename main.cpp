#include <igl/cat.h>
#include <igl/edge_lengths.h>
#include <igl/parula.h>
#include <igl/jet.h>
#include <igl/per_edge_normals.h>
#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>
#include <igl/point_mesh_squared_distance.h>
#include <igl/signed_distance.h>
#include <igl/flood_fill.h>
#include <igl/triangle/triangulate.h>
#include <igl/decimate.h>
#include <igl/opengl/glfw/Viewer.h>
#include <iostream>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

using namespace Eigen;
using u32 = std::uint32_t;

void precomputeStructures(MatrixXd& V, MatrixXi F) {
    // Precompute signed distance AABB tree
    printf("Compute AABB tree\n");
    igl::AABB<MatrixXd, 3> tree;
    tree.init(V, F);

    // Precompute vertex, edge and face normals
    printf("Compute normals\n");

    // Precomputed normals for vertices, edges, faces
    MatrixXd VN, EN, FN;
    MatrixXi E; // edges
    VectorXi EMAP; // map faces to their "half"-edge
    igl::per_face_normals(V, F, FN);
    igl::per_vertex_normals(
        V, F, igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE, FN, VN);
    igl::per_edge_normals(
        V, F, igl::PER_EDGE_NORMALS_WEIGHTING_TYPE_UNIFORM, FN, EN, E, EMAP);
}

MatrixXd generateVolumePoints(u32* volDim, double* origin, double margin, double* step) {
    MatrixXd P(volDim[0] * volDim[1] * volDim[2], 3);
    printf("Generate volume points\n");
    for (u32 x = 0; x < volDim[0]; x++) {
        for (u32 y = 0; y < volDim[1]; y++) {
            for (u32 z = 0; z < volDim[2]; z++) {
                u32 row = z + y * volDim[2] + x * volDim[1] * volDim[2];
                P(row, 0) = origin[0] - margin + x * step[0];
                P(row, 1) = origin[1] - margin + y * step[1];
                P(row, 2) = origin[2] - margin + z * step[2];
            }
        }
    }
    return P;
}

float* reorderVolumePoints(float* data, u32* res) {
    // z major to x major order
    u32 size = res[0] * res[1] * res[2];
    float* xdata = new float[size];
    for (u32 z = 0; z < res[2]; z++) {
        for (u32 y = 0; y < res[1]; y++) {
            for (u32 x = 0; x < res[0]; x++) {
                u32 i_old = z + y * res[2] + x * res[1] * res[2];
                u32 i_new = x + y * res[0] + z * res[0] * res[1];
                printf("%u -> %u (of %u)\n", i_old, i_new, size);
                xdata[i_new] = data[i_old];
            }
        }
    }
    return xdata;
}

void triangulateQuad(double xMin, double yMin, double zMin, double xMax, double yMax, double zMax) {
    // Triangulate 2d quad
    MatrixXd quad(4, 2);
    quad(0, 0) = xMin;
    quad(0, 1) = yMin;
    quad(1, 0) = xMax;
    quad(1, 1) = yMin;
    quad(2, 0) = xMin;
    quad(2, 1) = yMax;
    quad(3, 0) = xMax;
    quad(3, 1) = yMax;
    Eigen::MatrixXi quadE(4, 2);
    Eigen::MatrixXd quadH(0, 2);
    quadE(0, 0) = 0; quadE(0, 1) = 1;
    quadE(1, 0) = 1; quadE(1, 1) = 2;
    quadE(2, 0) = 2; quadE(2, 1) = 3;
    quadE(3, 0) = 3; quadE(3, 1) = 0;
    printf("Start triangulation\n");
    MatrixXd planeV;
    MatrixXi planeF;
    igl::triangle::triangulate(quad, quadE, quadH, "a0.5q", planeV, planeF);
}

VectorXf computeSignedDistance(MatrixXd& P, MatrixXd& V, MatrixXi F) {
    VectorXf d;
    VectorXi I;
    MatrixXd N, C;

    printf("Compute signed distance\n");

    // For watertight mesh use pseudonormal for signing
    //igl::signed_distance_pseudonormal(P, V, F, tree, FN, VN, EN, EMAP, d, I, C, N);

    // alternative way, without precomputed normals
    igl::SignedDistanceType type = igl::SIGNED_DISTANCE_TYPE_WINDING_NUMBER;
    igl::signed_distance(P, V, F, type, d, I, C, N); // had to abort after it ran for more than a day
    return d;
}

float* openVolume(std::string name, u32* res) {
    std::ifstream rawFile(name + ".raw", std::ios::binary);
    float* data = new float[res[0] * res[1] * res[2]];
    rawFile.read(reinterpret_cast<char*>(data), res[0] * res[1] * res[2] * sizeof(float));
    return data;
}

void saveVolume(std::string name, const void* data, float min, float max, u32* res, double step[3], bool zMajor = true) {

    // Write inviwo file format

    std::ofstream datFile(name + ".dat");
    datFile << "Rawfile: " + name + ".raw" << std::endl;

    if (zMajor) // need to swizzle because of voxel order
        datFile << "Resolution: " << res[2] << " " << res[1] << " " << res[0] << std::endl;
    else
        datFile << "Resolution: " << res[0] << " " << res[1] << " " << res[2] << std::endl;
    //TODO Re-order voxels to be x-major (in computeVolumePoints)
    //TODO add function to read the raw file to avoid recompute signed distances when just wanting to change layout

    datFile << "Format: Float32" << std::endl;
    datFile << "DataRange: " << min << " " << max << std::endl;

    // set basis so that it transforms texture coords to real units
    if (zMajor) {
        datFile << "BasisVector1: " << (double)res[2] * step[2] << " 0 0" << std::endl;
        datFile << "BasisVector2: 0 " << (double)res[1] * step[1] << " 0" << std::endl;
        datFile << "BasisVector3: 0 0 " << (double)res[0] * step[0] << std::endl;
    }
    else {
        datFile << "BasisVector1: " << (double)res[0] * step[0] << " 0 0" << std::endl;
        datFile << "BasisVector2: 0 " << (double)res[1] * step[1] << " 0" << std::endl;
        datFile << "BasisVector3: 0 0 " << (double)res[2] * step[2] << std::endl;
    }
    //datFile << "Offset: 0 0 0" << std::endl; // automatically chosen by inviwo

    std::ofstream rawFile(name + ".raw", std::ios::binary);
    rawFile.write(reinterpret_cast<const char*>(data), res[0] * res[1] * res[2] * sizeof(float));
}





























// This function is the actual main function.
void processMesh(aiMesh* mesh) {
    printf("Process mesh \"%s\"\n\
    Vertices: %u\n\
    Faces: %u\n\
    AABB: [(%f,%f,%f), (%f,%f,%f)]\n",
        mesh->mName.C_Str(), mesh->mNumVertices, mesh->mNumFaces,
        mesh->mAABB.mMin.x, mesh->mAABB.mMin.y, mesh->mAABB.mMin.z,
        mesh->mAABB.mMax.x, mesh->mAABB.mMax.y, mesh->mAABB.mMax.z);

    printf("Create V,F matrices\n");

    MatrixXd V(mesh->mNumVertices, 3);
    for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
        V(i, 0) = mesh->mVertices[i].x;
        V(i, 1) = mesh->mVertices[i].y;
        V(i, 2) = mesh->mVertices[i].z;
    }

    MatrixXi F(mesh->mNumFaces, 3);
    for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
        F(i, 0) = mesh->mFaces[i].mIndices[0];
        F(i, 1) = mesh->mFaces[i].mIndices[1];
        F(i, 2) = mesh->mFaces[i].mIndices[2];
    }

    // These experiments are not needed anymore, but left here for documentation.

    //precomputeStructures(V, F);

    //printf("Decimate mesh");
    //VectorXi J;
    //if (igl::decimate(V, F, 10000, U, G, J)) printf("ok"); // igl::decimate took hours, had to abort

    //printf("Tetrahedralize mesh");
    //MatrixXd TV; MatrixXi TT; MatrixXi TF;
    //igl::copyleft::tetgen::tetrahedralize(V, F, "pq1.414", TV, TT, TF);

    // Can use triangulated quad to visualize e.g. signed distances computed at its vertices
    //triangulateQuad(mesh->mAABB.mMin.x, mesh->mAABB.mMin.y, mesh->mAABB.mMin.z, 
    //    mesh->mAABB.mMax.x, mesh->mAABB.mMax.y, mesh->mAABB.mMax.z);

    // Define volume
    double origin[3];
    origin[0] = mesh->mAABB.mMin.x;
    origin[1] = mesh->mAABB.mMin.y;
    origin[2] = mesh->mAABB.mMin.z;
    double extent[3];
    extent[0] = double(mesh->mAABB.mMax.x - mesh->mAABB.mMin.x);
    extent[1] = double(mesh->mAABB.mMax.y - mesh->mAABB.mMin.y);
    extent[2] = double(mesh->mAABB.mMax.z - mesh->mAABB.mMin.z);
    const double margin = 5.0;
    const double scale = 0.5; // CONTROL COMPUTATION SPEED HERE!
    u32 res[3];
    res[0] = u32((extent[0] + margin * 2) * scale);
    res[1] = u32((extent[1] + margin * 2) * scale);
    res[2] = u32((extent[2] + margin * 2) * scale);
    double step[3];
    step[0] = (extent[0] + margin * 2) / double(res[0]);
    step[1] = (extent[1] + margin * 2) / double(res[1]);
    step[2] = (extent[2] + margin * 2) / double(res[2]);

    MatrixXd P = generateVolumePoints(res, origin, margin, step);
    // note: points in P are in z-major order

    // Compute signed distance
    VectorXf d = computeSignedDistance(P, V, F);

    printf("Flood fill NaNs\n");
    igl::flood_fill(Vector3i(res[2], res[1], res[0]), d);

    // Uncomment to reorder voxels in a previously saved volume.
    // Note the hardcoded filenames.
    //float* data = openVolume("phantom-sdf-volume", res);
    //float* xdata = reorderVolumePoints(data, res);
    //float min = std::numeric_limits<float>::infinity();
    //float max = -std::numeric_limits<float>::infinity();
    //for (u32 i = 0; i < res[0] * res[1] * res[2]; i++) {
    //    if (xdata[i] < min) min = xdata[i];
    //    if (xdata[i] > max) max = xdata[i];
    //}
    //saveVolume("phantom-sdf-volume-x", xdata, min, max, res, step, false);

    const float* sdfVolume = d.data(); // order refers to P

    saveVolume("phantom-sdf-volume", sdfVolume, d.minCoeff(), d.maxCoeff(), res, step);
}




























void traverseScene(const aiScene* scene, aiNode* node) {
    printf("Look at node \"%s\"\n", node->mName.C_Str());

    for (unsigned int i = 0; i < node->mNumMeshes; i++)
        processMesh(scene->mMeshes[node->mMeshes[i]]);

    for (unsigned int i = 0; i < node->mNumChildren; i++)
        traverseScene(scene, node->mChildren[i]);
}

int main(int nargs, char** args)
{
    using namespace Eigen;
    using namespace std;

    if (nargs < 2) {
        std::cout << "Convert mesh to SDF on regular grid in 3D." << std::endl;
        std::cout << "Please pass mesh file name." << std::endl;
        return 1;
    }

    Assimp::Importer importer;

    const aiScene* scene = importer.ReadFile(args[1],
        aiProcess_Triangulate |
        aiProcess_SortByPType |
        aiProcess_GenBoundingBoxes);

    if (!scene) {
        printf("%s\n", importer.GetErrorString());
        return 1;
    }

    traverseScene(scene, scene->mRootNode);

    // See processMesh() for actual work
}