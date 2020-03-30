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
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <iostream>
#include <array>

using namespace Eigen;
using u32 = std::uint32_t;

// Extract first mesh from scene.
aiMesh* assimpMesh(const aiScene* scene, aiNode* node = 0) {
    if (!node) node = scene->mRootNode;

    printf("Assimp found \"%s\"\n", node->mName.C_Str());

    for (unsigned int i = 0; i < node->mNumMeshes; i++)
        return scene->mMeshes[node->mMeshes[i]];

    for (unsigned int i = 0; i < node->mNumChildren; i++)
        assimpMesh(scene, node->mChildren[i]);
}

// Create libigl V, F matrices from Assimp mesh.
std::pair<MatrixXd, MatrixXi> assimpToIGLMesh(aiMesh* mesh) {
    printf("Assimp found \"%s\"\n\
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

    return { V, F };
}

// Fill space of given extents with points using regular steps.
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

// z major to x major order
float* reorderVolumePoints(float* data, u32* res) {
    u32 size = res[0] * res[1] * res[2];
    float* xdata = new float[size];
    for (u32 z = 0; z < res[2]; z++) {
        for (u32 y = 0; y < res[1]; y++) {
            for (u32 x = 0; x < res[0]; x++) {
                u32 i_old = z + y * res[2] + x * res[1] * res[2];
                u32 i_new = x + y * res[0] + z * res[0] * res[1];
                xdata[i_new] = data[i_old];
            }
        }
    }
    return xdata;
}

// Regular grid with explicit point storage.
struct LinSpace3D {
    MatrixXd P;
    std::array<u32, 3> res;
    std::array<double, 3> step;
};

// Generate regular grid points from Assimp AABB.
// One point per volume unit per default.
// Pass scale to control number of points.
// Pass margin to add space around.
// Points in z-major order.
LinSpace3D assimpToGrid(aiAABB aabb, const double scale = 1.0, const double margin = 5.0) {
    std::array<u32, 3> res;
    double origin[3];
    double extent[3];
    std::array<double,3> step;

    // Define volume
    origin[0] = aabb.mMin.x;
    origin[1] = aabb.mMin.y;
    origin[2] = aabb.mMin.z;
    extent[0] = double(aabb.mMax.x - aabb.mMin.x);
    extent[1] = double(aabb.mMax.y - aabb.mMin.y);
    extent[2] = double(aabb.mMax.z - aabb.mMin.z);
    res[0] = u32((extent[0] + margin * 2) * scale);
    res[1] = u32((extent[1] + margin * 2) * scale);
    res[2] = u32((extent[2] + margin * 2) * scale);
    step[0] = (extent[0] + margin * 2) / double(res[0]);
    step[1] = (extent[1] + margin * 2) / double(res[1]);
    step[2] = (extent[2] + margin * 2) / double(res[2]);

    std::cout << "Grid res = " << res[0] << " " << res[1] << " " << res[2] << std::endl;

    return { generateVolumePoints(res.data(), origin, margin, step.data()), res, step };
}

// Sample signed distances to mesh surface at grid points.
VectorXf sampleSDF(MatrixXd P, std::array<u32, 3> res, MatrixXd V, MatrixXi F) {
    VectorXf d;
    VectorXi I;
    MatrixXd N, C;

    printf("Compute signed distance\n");
    igl::SignedDistanceType type = igl::SIGNED_DISTANCE_TYPE_WINDING_NUMBER;
    igl::signed_distance(P, V, F, type, d, I, C, N);

    printf("Flood fill to eliminate possible NaNs\n");
    igl::flood_fill(Vector3i(res[2], res[1], res[0]), d);

    return d;
}

// Get data from Inviwo volume file. You have to know the volume resolution.
float* openVolume(std::string name, u32* res) {
    std::ifstream rawFile(name + ".raw", std::ios::binary);
    float* data = new float[res[0] * res[1] * res[2]];
    rawFile.read(reinterpret_cast<char*>(data), res[0] * res[1] * res[2] * sizeof(float));
    return data;
}

// Save volume in Inviwo file format.
void saveVolume(const std::string name, const void* data, const float min, const float max, const u32* res, const double step[3], const bool zMajor = true) {
    std::ofstream datFile(name + ".dat");
    datFile << "Rawfile: " + name + ".raw" << std::endl;
    if (zMajor)
        datFile << "Resolution: " << res[2] << " " << res[1] << " " << res[0] << std::endl;
    else
        datFile << "Resolution: " << res[0] << " " << res[1] << " " << res[2] << std::endl;

    datFile << "Format: Float32" << std::endl;
    datFile << "DataRange: " << min << " " << max << std::endl;
    // Basis transforms tex coords to real units
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
    // Offset: 0 0 0 automatically chosen by inviwo
    std::ofstream rawFile(name + ".raw", std::ios::binary);
    rawFile.write(reinterpret_cast<const char*>(data), res[0] * res[1] * res[2] * sizeof(float));
}

// Save sampled SDF to Inviwo volume for visualization.
void sdfToInviwoVolume(const VectorXf d, const std::array<u32, 3> res, const std::array<double, 3> step, const std::string file) {
    std::cout << "Save Inviwo volume to " << file << ".dat" << std::endl;
    saveVolume(file, d.data(), d.minCoeff(), d.maxCoeff(), res.data(), step.data());
}

// Flip voxel order. Try when visualization looks wrong.
void reorderInviwoVoxels(std::string file, u32* res, double* step) {

    float* data = openVolume(file, res);
    float* xdata = reorderVolumePoints(data, res);
    float min = std::numeric_limits<float>::infinity();
    float max = -std::numeric_limits<float>::infinity();
    for (u32 i = 0; i < res[0] * res[1] * res[2]; i++) {
        if (xdata[i] < min) min = xdata[i];
        if (xdata[i] > max) max = xdata[i];
    }
    saveVolume(file + "-reordered", xdata, min, max, res, step, false);
}

int cleaner_main(int nargs, char** args) {
    if (nargs < 2) {
        std::cout << "Convert mesh to SDF on regular grid in 3D." << std::endl;
        std::cout << "Please pass mesh file name." << std::endl;
        std::cout << "Optionally pass grid scale." << std::endl;
        return 1;
    }

    Assimp::Importer i;
    const auto s = i.ReadFile(args[1],
        aiProcess_Triangulate |
        aiProcess_SortByPType |
        aiProcess_GenBoundingBoxes);

    if (!s) {
        std::cout << i.GetErrorString() << std::endl;
        return 1;
    }

    const auto mesh = assimpMesh(s);

    const auto VF = assimpToIGLMesh(mesh);
    const auto grid = assimpToGrid(mesh->mAABB, nargs > 2 ? atof(args[2]) : 1);
    const auto sdf = sampleSDF(grid.P, grid.res, VF.first, VF.second);

    sdfToInviwoVolume(sdf, grid.res, grid.step, "VOLUME");

    return 0;
}