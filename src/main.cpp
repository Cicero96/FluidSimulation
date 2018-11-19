// This example is heavily based on the tutorial at https://open.gl

// OpenGL Helpers to reduce the clutter
#include "Helpers.h"

// GLFW is necessary to handle the OpenGL context
#include <GLFW/glfw3.h>

// Linear Algebra Library
#include <Eigen/Core>
#include <Eigen/Dense>

// Timer
#include <chrono>

#include <cmath>
#include <vector>
#include <map>
#include <thread>
#include <fstream>
#include <iostream>
#include <ctime>
#include <cstdlib>

using namespace std;
using namespace Eigen;

float radius = 5;
Vector3f cameraPosition = Vector3f(0, 0, radius);
Vector3f lightPosition = Vector3f(0, 2, 2);
Vector3f lightColor = Vector3f(1.0f, 1.0f, 1.0f);
int theta = 0, phi = 0;
double pi = 3.14159265359;
float l = -1, r = 1, t = 1, b = -1, n = 3, f = 10;
// bool WIREFRAME = false;
bool PERSPECTIVE = true;
bool ELEMENT = true;
bool ANIMATION = false;
enum EDITMODE {FREE, INSERT, TRANSLATION, DELETE};
EDITMODE editMode = FREE;

double startXPos = -2, startYPos = -2;
float aspectRatio = -1;

MatrixXf MModel(4, 4);
MatrixXf MCam(4, 4);
MatrixXf MProj(4, 4);
MatrixXf MOrth(4, 4);
MatrixXf Sceneview(4, 4);

bool DEBUG = false;

class BezierCurve
{
public:
    MatrixXf V;
    MatrixXf tmpV;
    Matrix4f view;
    VertexBufferObject VBO;
    VertexArrayObject VAO;
    int targetCurveVertex = -1;

    BezierCurve()
    {
        V.resize(3, 0);

        view << 1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1;

        VAO.init();
        VAO.bind();

        VBO.init();
        VBO.update(V);
    }
};

class Element
{
public:
    MatrixXf V;
    MatrixXf N;
    MatrixXf NF;
    MatrixXi E;
    Matrix4f view;
    Vector3d color;
    Vector3d selectedColor;
    VertexBufferObject VBO;
    VertexBufferObject NBO;
    ElementBufferObject EBO;
    VertexArrayObject VAO;
    float r, l, t, b, n, f;
    float baryCenterX, baryCenterY, baryCenterZ;
    bool wireframe = false;
    bool flat = false;
    BezierCurve bezierCurve;

    Element()
    {
    }

    Element(MatrixXf _v, MatrixXi _e)
    {
        V = _v;
        E = _e;

        srand(time(NULL));
        for (int i = 0; i < 3; i ++)
        {
            color(i) = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/1));
        }
        selectedColor = Vector3d(0.5, 0, 0);

        r = V.row(0).maxCoeff();
        l = V.row(0).minCoeff();
        t = V.row(1).maxCoeff();
        b = V.row(1).minCoeff();
        n = V.row(2).maxCoeff();
        f = V.row(2).minCoeff();

        baryCenterX = V.row(0).mean();
        baryCenterY = V.row(1).mean();
        baryCenterZ = V.row(2).mean();

        // Scale everything to [-0.5, 0.5]
        view << 1.0 / (r - l), 0, 0, -(r + l) / (2 * (r - l)),
                0, 1.0 / (t - b), 0, -(t + b) / (2 * (t - b)),
                0, 0, 1.0 / (n - f), -(n + f) / (2 * (n - f)), 
                0, 0, 0, 1;
        
        AppendAverageNormals();

        bezierCurve = BezierCurve();

        VAO.init();
        VAO.bind();

        VBO.init();
        VBO.update(V);

        NBO.init();
        NBO.update(N);

        EBO.init();
        EBO.update(E);
    }

    void AppendAverageNormals()
    {
        cout << "In AppendAverageNormals" << endl;
        int row = V.rows(), col = V.cols();
        N = MatrixXf::Zero(row, col);
        NF = MatrixXf::Zero(row, E.cols());
        for (int i = 0; i < (int)E.cols(); i ++)
        {
            int indexOne = E(0, i), indexTwo = E(1, i), indexThree = E(2, i);
            Vector3f a = V.col(indexOne), b = V.col(indexTwo), c = V.col(indexThree);
            Vector3f ab = b - a;
            Vector3f ac = c - a;
            Vector3f surfaceNormal = ab.cross(ac).normalized();
            NF.col(i) = surfaceNormal;
            for (int j = 0; j < 3; j ++)
            {
                N.col(E(j, i)) += surfaceNormal;
            }
        }

        for (int i = 0; i < col; i ++)
        {
            N.col(i) = N.col(i).normalized();
        }
    }

    bool Intersect(Vector4f clickPoint, double& dist)
    {
        Vector3f rayOrigion;
        Vector3f rayDirection;
        if (PERSPECTIVE)
        {
            rayOrigion = Vector3f(cameraPosition(0), cameraPosition(1), cameraPosition(2));
            rayDirection << clickPoint(0) - rayOrigion(0),
                        clickPoint(1) - rayOrigion(1),
                        clickPoint(2) - rayOrigion(2);
        }
        else
        {
            rayOrigion = Vector3f(clickPoint(0), clickPoint(1), clickPoint(2));
            rayDirection << -cameraPosition(0), -cameraPosition(1), -cameraPosition(2);
        }
        rayDirection = rayDirection.normalized();

        dist = 1e10;
        int size = E.cols();
        for (int i = 0; i < size; i ++)
        {
            Vector4f tmpA = Vector4f(V(0, E(0, i)), V(1, E(0, i)), V(2, E(0, i)), 1);
            Vector4f tmpB = Vector4f(V(0, E(1, i)), V(1, E(1, i)), V(2, E(1, i)), 1);
            Vector4f tmpC = Vector4f(V(0, E(2, i)), V(1, E(2, i)), V(2, E(2, i)), 1);
            tmpA = view * tmpA;
            tmpB = view * tmpB;
            tmpC = view * tmpC;
            Vector3f a = Vector3f(tmpA(0), tmpA(1), tmpA(2));
            Vector3f b = Vector3f(tmpB(0), tmpB(1), tmpB(2));
            Vector3f c = Vector3f(tmpC(0), tmpC(1), tmpC(2));
            Matrix3f m = Matrix3f::Zero(3, 3);
            Vector3f ab = a - b;
            Vector3f ac = a - c;
            Vector3f ae = a - rayOrigion;

            // Cramer's rule is faster.
            float eiMinusHf = ac(1) * rayDirection(2) - ac(2) * rayDirection(1);
            float gfMinusDi = ac(2) * rayDirection(0) - ac(0) * rayDirection(2);
            float dhMinusEg = ac(0) * rayDirection(1) - ac(1) * rayDirection(0);
            float akMinusJb = ab(0) * ae(1) - ab(1) * ae(0);
            float jcMinusAl = ab(2) * ae(0) - ab(0) * ae(2);
            float blMinusKc = ab(1) * ae(2) - ab(2) * ae(1);
            float M = ab(0) * eiMinusHf + ab(1) * gfMinusDi + ab(2) * dhMinusEg;

            float t = -1 * (ac(2) * akMinusJb + ac(1) * jcMinusAl + ac(0) * blMinusKc) / M;
            if (t >= 0 && t < dist)
            {
                float gama = (rayDirection(2) * akMinusJb + rayDirection(1) * jcMinusAl + rayDirection(0) * blMinusKc) / M;
                if (gama >= 0 && gama <= 1)
                {
                    float beta = (ae(0) * eiMinusHf + ae(1) * gfMinusDi + ae(2) * dhMinusEg) / M;
                    if (beta >= 0 && beta <= 1 - gama)
                    {
                        dist = t;
                    }
                }
            }
        }
        
        return dist < 1e10;
    }
};

vector<Element*> elements = vector<Element*>();
int targetElement = -1;

void ConvertScreenPosToWorldPos(GLFWwindow* window, double xpos, double ypos, double* xworld, double* yworld)
{
    // Get the size of the window
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // Convert screen position to world coordinates
    Vector4f p_screen(xpos, height - 1 - ypos, 0, 1);
    Vector4f p_canonical((p_screen[0] / width) * 2 - 1, (p_screen[1] / height) * 2 - 1, 0, 1);
    Vector4f p_world = Sceneview.inverse() * p_canonical;

    *xworld = p_world[0];
    *yworld = p_world[1];
}

void GetCursorPos(GLFWwindow* window, double* xworld, double* yworld)
{
    // Get the position of the mouse in the window
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);

    ConvertScreenPosToWorldPos(window, xpos, ypos, xworld, yworld);
}

void UpdateCameraMatrix()
{
    Vector3f cameraDirection = cameraPosition.normalized();
    Vector3f cameraRight;

    if (phi < 90)
    {
        Vector3f viewUp = Vector3f(0, 1, 0);
        cameraRight = viewUp.cross(cameraDirection).normalized();
    }
    else if (phi == 90)
    {
        Vector3f viewUp = Vector3f(0, 1, 0);
        Vector3f xz;
        xz << sin((double)theta / 180 * pi), 0, cos((double)theta / 180 * pi);
        cameraRight = viewUp.cross(xz).normalized();
    }
    else if (phi > 90 && phi < 270)
    {
        Vector3f viewUp = Vector3f(0, -1, 0);
        cameraRight = viewUp.cross(cameraDirection).normalized();
    }
    else if (phi == 270)
    {
        Vector3f viewUp = Vector3f(0, 1, 0);
        Vector3f xz;
        xz << sin((double)theta / 180 * pi), 0, cos((double)theta / 180 * pi);
        cameraRight = viewUp.cross(xz).normalized();
    }
    else
    {
        Vector3f viewUp = Vector3f(0, 1, 0);
        cameraRight = viewUp.cross(cameraDirection).normalized();
    }
    Vector3f cameraUp = cameraDirection.cross(cameraRight).normalized();

    MCam << cameraRight(0), cameraRight(1), cameraRight(2), 0,
            cameraUp(0), cameraUp(1), cameraUp(2), 0,
            cameraDirection(0), cameraDirection(1), cameraDirection(2), 0,
            0, 0, 0, 1;
    
    Matrix4f tmp;
    tmp << 1, 0, 0, -cameraPosition(0),
           0, 1, 0, -cameraPosition(1),
           0, 0, 1, -cameraPosition(2),
           0, 0, 0, 1;
    
    MCam = MCam * tmp;

    if (DEBUG)
    {
        cout << "In UpdateCameraMatrix" << endl;
        cout << MCam << endl;
    }
}

void UpdateOrthographicProjection()
{
    MOrth << 2.0 / (r - l), 0, 0, -(r + l) / (r - l),
             0, 2.0 / (t - b), 0, -(t + b) / (t - b),
             0, 0, -2 / (f - n), -(f + n) / (f - n), 
             0, 0, 0, 1;
    
    if (DEBUG)
    {
        cout << "In UpdateOrthographicProjection" << endl;
        cout << MOrth << endl;
    }
}

void UpdatePerspectiveProjection()
{
    MProj << 2 * n / (r - l), 0, (r + l) / (r - l), 0,
             0, 2 * n / (t - b), (t + b) / (t - b), 0, 
             0, 0, -(f + n) / (f - n), -2 * (f * n) / (f - n), 
             0, 0, -1, 0;

    if (DEBUG)
    {
        cout << "In UpdatePerspectiveProjection" << endl;
        cout << MProj << endl;
    }
}

void UpdateSceneView(GLFWwindow* window)
{
    int width, height;
    glfwGetWindowSize(window, &width, &height);
    float aspect_ratio = float(height)/float(width);

    if (aspectRatio == -1)
    {
        Sceneview(0, 0) *= aspect_ratio;
        aspectRatio = aspect_ratio;
    }
    else if (aspect_ratio != aspectRatio)
    {
        cout << "here" << endl;
        Sceneview(0, 0) /= aspectRatio;
        Sceneview(0, 0) *= aspect_ratio;
        aspectRatio = aspect_ratio;
    }
}

void adjustCamera(int alpha, int beta)
{
    theta = (theta + alpha + 360) % 360;
    phi = (phi + beta + 360) % 360;
    cout << "theta = " << theta << ", phi = " << phi << endl;
    float x, y, z;
    y = radius * sin((double)phi / 180 * pi);
    x = radius * cos((double)phi / 180 * pi) * sin((double)theta / 180 * pi);
    z = radius * cos((double)phi / 180 * pi) * cos((double)theta / 180 * pi);
    cameraPosition << x, y, z;

    UpdateCameraMatrix();

    cout << cameraPosition << endl;
}

void ReadOffFile(const string fileName, MatrixXf& vertices, MatrixXi& index)
{
    vertices.resize(3, 502);
    index.resize(3, 1000);
    char data[100];
    int verticesNum, facesNum, edgesNum;

    cout << "Start reading file: " << fileName << endl;
    ifstream in(fileName);
    
    in >> data;
    in >> verticesNum >> facesNum >> edgesNum;
    float a, b, c;
    for (int i = 0; i < verticesNum; i ++)
    {
        in >> vertices(0, i) >> vertices(1, i) >> vertices(2, i);
    }

    int num;
    for (int i = 0; i < facesNum; i++)
    {
        in >> num;
        in >> index(0, i) >> index(1, i) >> index(2, i);
    }

    in.close();
    cout << "Reading finished" << endl;
}

bool FindLatestElement(double xPos, double yPos)
{
    Vector4f clickPoint = Vector4f(xPos, yPos, -n, 1);
    // if (PERSPECTIVE)
    // {
    //     clickPoint = MCam.inverse() * MProj.inverse() * clickPoint;
    // }
    // else
    // {
    //     clickPoint = MCam.inverse() * MOrth.inverse() * clickPoint;
    // }
    clickPoint = MCam.inverse() * clickPoint;

    int size = elements.size();
    double d = 1e10;

    // Check with each object, and find the shortest distance
    for (int i = 0; i < size; i ++)
    {
        double tmpDist = 1e10;
        if (elements[i]->Intersect(clickPoint, tmpDist))
        {
            if (tmpDist < d)
            {
                d = tmpDist;
                targetElement = i;
            }
        }
    }

    return d < 1e10;
}

void ScaleElement(float s)
{
    Element* cur = elements[targetElement];
    float baryCenterX = cur->baryCenterX, baryCenterY = cur->baryCenterY, baryCenterZ = cur->baryCenterZ;
    Vector4f tmp;
    tmp << baryCenterX, baryCenterY, baryCenterZ, 1;
    tmp = (cur->view) * tmp;
    baryCenterX = tmp(0);
    baryCenterY = tmp(1);
    baryCenterZ = tmp(2);

    Matrix4f tranToward;
    tranToward << 1, 0, 0, -baryCenterX,
            0, 1, 0, -baryCenterY,
            0, 0, 1, -baryCenterZ,
            0, 0, 0, 1;

    Matrix4f scale;
    scale << s, 0, 0, 0,
             0, s, 0, 0,
             0, 0, s, 0,
             0, 0, 0, 1;
    
    Matrix4f tranBack;
    tranBack << 1, 0, 0, baryCenterX,
                0, 1, 0, baryCenterY,
                0, 0, 1, baryCenterZ,
                0, 0, 0, 1;
    
    (cur->view) = tranBack * scale * tranToward * (cur->view);
}

void RotateElement(float theta)
{
    Element* cur = elements[targetElement];
    float baryCenterX = cur->baryCenterX, baryCenterY = cur->baryCenterY, baryCenterZ = cur->baryCenterZ;
    Vector4f tmp;
    tmp << baryCenterX, baryCenterY, baryCenterZ, 1;
    tmp = (cur->view) * tmp;
    baryCenterX = tmp(0);
    baryCenterY = tmp(1);
    baryCenterZ = tmp(2);

    Matrix4f tranToward;
    tranToward << 1, 0, 0, -baryCenterX,
            0, 1, 0, -baryCenterY,
            0, 0, 1, -baryCenterZ,
            0, 0, 0, 1;

    Vector3f normalizedCam = cameraPosition.normalized();
    float u = normalizedCam(0), v = normalizedCam(1), w = normalizedCam(2);
    theta = theta / 180 * pi;
    Matrix4f scale;
    scale << u*u+(1-u*u)*cos(theta), u*v*(1-cos(theta))-w*sin(theta), u*w*(1-cos(theta))+v*sin(theta), 0,
             u*v*(1-cos(theta))+w*sin(theta), v*v+(1-v*v)*cos(theta), v*w*(1-cos(theta))-u*sin(theta), 0,
             u*w*(1-cos(theta))-v*sin(theta), v*w*(1-cos(theta))+u*sin(theta), w*w+(1-w*w)*cos(theta), 0,
             0, 0, 0, 1;
    
    Matrix4f tranBack;
    tranBack << 1, 0, 0, baryCenterX,
                0, 1, 0, baryCenterY,
                0, 0, 1, baryCenterZ,
                0, 0, 0, 1;
    
    (cur->view) = tranBack * scale * tranToward * (cur->view);
}

void ZoomControl(float sx, float sy, float sz, float tx, float ty, float tz)
{
    Sceneview(0, 0) *= sx;
    Sceneview(1, 1) *= sy;
    Sceneview(2, 2) *= sz;
    Sceneview(0, 3) += tx;
    Sceneview(1, 3) += ty;
    Sceneview(2, 3) += tz;
}

void DrawCruves(BezierCurve* curve)
{
    MatrixXf I;
    I.resize(3, 101);
    for (int i = 0; i <= 100; i++)
    {
        double t = i / 100.0;
        double s = 1 - t;
        int size = curve->V.cols();
        vector<Vector3f> buffer = vector<Vector3f>();
        for (int j = 0; j < size; j ++)
        {
            buffer.push_back((curve->V).col(j));
        }
        for (int j = 0; j < size - 1; j ++)
        {
            for (int k = 0; k < size - 1 - j; k ++)
            {
                buffer[k] = buffer[k] * s + buffer[k + 1] * t;
            }
        }
        I.col(i) = buffer[0];
    }
    (curve->tmpV) = I;
}

bool IntersectTriangle(Vector3f rayDirection, Vector3f a, Vector3f b, Vector3f c, float &dist)
{
    dist = 1e10;
    Matrix3f m = Matrix3f::Zero(3, 3);
    Vector3f ab = a - b;
    Vector3f ac = a - c;
    Vector3f ae = a - cameraPosition;

    // Cramer's rule is faster.
    float eiMinusHf = ac(1) * rayDirection(2) - ac(2) * rayDirection(1);
    float gfMinusDi = ac(2) * rayDirection(0) - ac(0) * rayDirection(2);
    float dhMinusEg = ac(0) * rayDirection(1) - ac(1) * rayDirection(0);
    float akMinusJb = ab(0) * ae(1) - ab(1) * ae(0);
    float jcMinusAl = ab(2) * ae(0) - ab(0) * ae(2);
    float blMinusKc = ab(1) * ae(2) - ab(2) * ae(1);
    float M = ab(0) * eiMinusHf + ab(1) * gfMinusDi + ab(2) * dhMinusEg;

    float t = -1 * (ac(2) * akMinusJb + ac(1) * jcMinusAl + ac(0) * blMinusKc) / M;
    if (t >= 0 && t < dist)
    {
        float gama = (rayDirection(2) * akMinusJb + rayDirection(1) * jcMinusAl + rayDirection(0) * blMinusKc) / M;
        if (gama >= 0 && gama <= 1)
        {
            float beta = (ae(0) * eiMinusHf + ae(1) * gfMinusDi + ae(2) * dhMinusEg) / M;
            if (beta >= 0 && beta <= 1 - gama)
            {
                dist = t;
            }
        }
    }

    return dist < 1e10;
}

void ExportSVG(GLFWwindow* window)
{
    ofstream out("../result/opengl.svg");
    int width, height;
    glfwGetWindowSize(window, &width, &height);
    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width
        << "\" height=\"" << height << "\">" << endl;
    out << "<rect x=\"0\" y=\"0\" width=\"" << width << "\" height=\""
        << height << "\" style=\"fill:rgb(128,128,128);\" />" << endl;

    for (int i = 0; i < elements.size(); i ++)
    {
        int size = elements[i]->E.cols();
        for (int j = 0; j < size; j ++)
        {
            vector<float> vertexs = vector<float>();
            vector<Vector3f> vertexsV = vector<Vector3f>();
            vector<Vector3f> colors = vector<Vector3f>();

            // Get vertexs coordinates in world and screen
            for (int k = 0; k < 3; k ++)
            {
                Vector4f tmp;
                tmp << (elements[i]->V).col(elements[i]->E(k, j))(0), 
                       (elements[i]->V).col(elements[i]->E(k, j))(1),
                       (elements[i]->V).col(elements[i]->E(k, j))(2),
                       1;
                tmp = (elements[i]->view) * tmp;
                vertexsV.push_back(Vector3f(tmp(0), tmp(1), tmp(2)));
                if (PERSPECTIVE)
                {
                    tmp = MCam * tmp;
                    tmp = Sceneview * MProj * -(1 / tmp(2)) * tmp;
                }
                else
                {
                    tmp = Sceneview * MOrth * MCam * tmp;
                }
                vertexs.push_back((tmp(0) + 1) / 2 * width);
                vertexs.push_back(height - ((tmp(1) + 1) / 2 * height));
            }
            
            // Check intersection. Ignore this triangle if any vertex is invisiable
            bool seen = true;
            Vector3f bPoint = (1.0 / 3) * (vertexsV[0] + vertexsV[1] + vertexsV[2]); 
            Vector3f rayDirection = (bPoint - cameraPosition).normalized();
            float targetDist = (bPoint - cameraPosition).norm();

            for (int e = 0; e < elements.size(); e ++)
            {
                int tmpSize = elements[e]->E.cols();
                for (int ii = 0; ii < tmpSize; ii ++)
                {
                    Vector4f tmpA = Vector4f(elements[e]->V(0, elements[e]->E(0, ii)), elements[e]->V(1, elements[e]->E(0, ii)), elements[e]->V(2, elements[e]->E(0, ii)), 1);
                    Vector4f tmpB = Vector4f(elements[e]->V(0, elements[e]->E(1, ii)), elements[e]->V(1, elements[e]->E(1, ii)), elements[e]->V(2, elements[e]->E(1, ii)), 1);
                    Vector4f tmpC = Vector4f(elements[e]->V(0, elements[e]->E(2, ii)), elements[e]->V(1, elements[e]->E(2, ii)), elements[e]->V(2, elements[e]->E(2, ii)), 1);
                    tmpA = elements[e]->view * tmpA;
                    tmpB = elements[e]->view * tmpB;
                    tmpC = elements[e]->view * tmpC;
                    Vector3f a = Vector3f(tmpA(0), tmpA(1), tmpA(2));
                    Vector3f b = Vector3f(tmpB(0), tmpB(1), tmpB(2));
                    Vector3f c = Vector3f(tmpC(0), tmpC(1), tmpC(2));
                    float dist = 0;
                    IntersectTriangle(rayDirection, a, b, c, dist);
                    // cout << "targetDist = " << targetDist << ", dist = " << dist << endl;
                    if (dist - targetDist < -0.00001)
                    {
                        seen = false;
                        break;
                    }
                }
                if (!seen)
                {
                    break;
                }
            }
            if (!seen)
            {
                continue;
            }

            // Get color for per vertex
            Vector3f ab = vertexsV[1] - vertexsV[0];
            Vector3f ac = vertexsV[2] - vertexsV[0];
            Vector3f surfaceNormal = ab.cross(ac).normalized();
            for (int k = 0; k < 3; k ++)
            {
                float ambientStrength = 0.1;
                Vector3f ambient = ambientStrength * lightColor;
                Vector3f lightDir = (lightPosition - vertexsV[k]).normalized();
                float diff = surfaceNormal.transpose() * lightDir;
                if (diff < 0)
                {
                    diff = 0;
                }
                else if (diff > 1)
                {
                    diff = 1;
                }
                Vector3f diffuse = diff * lightColor;
                float specularStrength = 0.5;
                Vector3f viewDir = (cameraPosition - vertexsV[k]).normalized();
                float dn = 2.0 * surfaceNormal.transpose() * lightDir;
                Vector3f reflectDir = -1 * lightDir + dn * surfaceNormal;
                float val = viewDir.transpose() * reflectDir;
                float spec = pow(max(val, (float)0), 32);
                Vector3f specular = specularStrength * spec * lightColor;
                Vector3f sum = ambient + diffuse + specular;
                if (i == targetElement)
                {
                    colors.push_back(255 * Vector3f(sum(0)*elements[i]->selectedColor(0), sum(1)*elements[i]->selectedColor(1), sum(2)*elements[i]->selectedColor(2)));
                }
                else
                {
                    colors.push_back(255 * Vector3f(sum(0)*elements[i]->color(0), sum(1)*elements[i]->color(1), sum(2)*elements[i]->color(2)));
                }
            }

            // Same code from last assignment
            // Create two triangle to simulate interpolation
            double dx = vertexs[2] - vertexs[4];
            double dy = vertexs[3] - vertexs[5];
            double u = (vertexs[0]  - vertexs[2]) * dx + (vertexs[1] - vertexs[3]) * dy;
            u = u / ((dx * dx) + (dy * dy));
            vertexs.push_back(vertexs[2] + u * dx);
            vertexs.push_back(vertexs[3] + u * dy);

            out << "<defs>\n<linearGradient id=\"gradient_" << i << "_" << j << "\" gradientUnits=\"userSpaceOnUse\" ";
            out << "x1=\"" << vertexs[2] << "\" y1=\"" << vertexs[3] << "\" "
                << "x2=\"" << vertexs[4] << "\" y2=\"" << vertexs[5] << "\">" << endl;
            out << "<stop offset=\"0%\" stop-color=\"rgb(" << colors[1][0] << "," << colors[1][1]
                << "," << colors[1][2] << ")\" />" << endl;
            out << "<stop offset=\"100%\" stop-color=\"rgb(" << colors[2][0] << "," << colors[2][1]
                << "," << colors[2][2] << ")\" />\n</linearGradient>" << endl;
            out << "<linearGradient id=\"fader_" << i << "_" << j << "\" gradientUnits=\"userSpaceOnUse\" ";
            out << "x1=\"" << vertexs[0] << "\" y1=\"" << vertexs[1] << "\" "
                << "x2=\"" << vertexs[6] << "\" y2=\"" << vertexs[7] << "\">" << endl;
            out << "<stop offset=\"0%\" stop-color=\"white\" />" << endl;
            out << "<stop offset=\"100%\" stop-color=\"black\" />\n</linearGradient>" << endl;
            out << "<mask id=\"mask_" << i << "_" << j << "\" maskUnits=\"userSpaceOnUse\" "
                << "maskContentUnits=\"maskContentUnits\">" << endl;
            out << "<polygon points=\"";
            for (int k = 0; k < 6; k += 2)
            {
                out << vertexs[k] << "," << vertexs[k + 1] << " ";
            }
            out << "\" style=\"fill:url(#fader_" << i << "_" << j << ");\" />\n</mask></defs>" << endl;

            out << "<polygon points=\"";
            for (int k = 0; k < 6; k += 2)
            {
                out << vertexs[k] << "," << vertexs[k + 1] << " ";
            }
            out << "\" style=\"stroke:black;fill:url(#gradient_" << i << "_" << j << ");\" />" << endl;

            out << "<polygon points=\"";
            for (int k = 0; k < 6; k += 2)
            {
                out << vertexs[k] << "," << vertexs[k + 1] << " ";
            }
            out << "\" style=\"stroke:black;fill:rgb(" << colors[0][0] << "," << colors[0][1]
                << "," << colors[0][2] << "); mask:url(#mask_" << i << "_" << j << ");\" />" << endl;
        }
    }

    out << "</svg>" << endl;
    out.close();
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (editMode == FREE)
    {
        if (action == GLFW_RELEASE)
        {
            printf("Mouse released in free mode\n");
            targetElement = -1;
        }
        else if (action == GLFW_PRESS)
        {
            printf("Mouse pressed in free mode\n");
        }
    }
    else if (editMode == TRANSLATION)
    {
        double xWorld, yWorld;
        GetCursorPos(window, &xWorld, &yWorld);
        if (button == GLFW_MOUSE_BUTTON_LEFT)
        {
            if (action == GLFW_PRESS)
            {
                printf("Mouse pressed in translation mode\n");
                if (startXPos == -2 && startYPos == -2)
                {
                    if (FindLatestElement(xWorld, yWorld))
                    {
                        printf("Find the element\n");
                        startXPos = xWorld;
                        startYPos = yWorld;
                    }
                    else
                    {
                        targetElement = -1;
                    }
                }
                else
                {
                    printf("This should not happen!\n");
                }
            }
            else if (action == GLFW_RELEASE)
            {
                if (editMode == TRANSLATION)
                {
                    printf("Mouse released in translation mode\n");
                }
                startXPos = -2;
                startYPos = -2;
            }
        }
    }
    else if (editMode == DELETE)
    {
        double xWorld, yWorld;
        GetCursorPos(window, &xWorld, &yWorld);
        if (button == GLFW_MOUSE_BUTTON_LEFT)
        {
            if (action == GLFW_RELEASE)
            {
                printf("Mouse released in delete mode\n");
                if (FindLatestElement(xWorld, yWorld))
                {
                    printf("Find the element, delete it\n");
                    elements.erase(elements.begin() + targetElement);
                    targetElement = -1;
                }
            }
            else if (action == GLFW_PRESS)
            {
                printf("Mouse pressed in delete mode\n");
            }
        }
    }
}

void MousePosCallback(GLFWwindow* window, double xpos, double ypos)
{
    if (editMode == TRANSLATION)
    {
        if (!(startXPos + 2 < 1e-5) && !(startYPos + 2 < 1e-5))
        {
            double endXPos, endYPos;
            GetCursorPos(window, &endXPos, &endYPos);

            cout << "dx = " << endXPos - startXPos
                 << ", dy = " << endYPos - startYPos << endl;

            Element* cur = elements[targetElement];
            cout << "Select element index = " << targetElement << endl;

            Vector4f tmpStart = Vector4f(startXPos, startYPos, -n, 1);
            if (PERSPECTIVE)
            {
                tmpStart = MCam.inverse() * ((f / n) * MProj.inverse() * tmpStart);
                // tmpStart = MCam.inverse() * ((f / n) * tmpStart);
            }
            else
            {
                tmpStart = MCam.inverse() * (MOrth.inverse() * tmpStart);
                // tmpStart = MCam.inverse() * (tmpStart);
            }
            Vector4f tmpEnd = Vector4f(endXPos, endYPos, -n, 1);
            if (PERSPECTIVE)
            {
                tmpEnd = MCam.inverse() * ((f / n) * MProj.inverse() * tmpEnd);
                // tmpEnd = MCam.inverse() * ((f / n) * tmpEnd);
            }
            else
            {
                tmpEnd = MCam.inverse() * (MOrth.inverse() * tmpEnd);
                // tmpEnd = MCam.inverse() * (tmpEnd);
            }

            Matrix4f tranToward;
            tranToward << 1, 0, 0, tmpEnd(0) - tmpStart(0),
                          0, 1, 0, tmpEnd(1) - tmpStart(1),
                          0, 0, 1, tmpEnd(2) - tmpStart(2),
                          0, 0, 0, 1;
            
            (cur->view) = tranToward * (cur->view);
            cout << cur->view << endl;

            startXPos = endXPos;
            startYPos = endYPos;
        }
    }
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    // Update the position of the first vertex if the keys 1,2, or 3 are pressed
    MatrixXf vertices;
    MatrixXi index;
    switch (key)
    {
        case GLFW_KEY_I:
            if (action == GLFW_RELEASE)
            {
                if (editMode != INSERT)
                {
                    cout << "Switch into insert mode" << endl;
                    editMode = INSERT;
                }
                else
                {
                    cout << "Exit insert mode" << endl;
                    editMode = FREE;
                }
            }
            break;
        case GLFW_KEY_O:
            if (action == GLFW_RELEASE)
            {
                if (editMode != TRANSLATION)
                {
                    cout << "Switch into translation mode" << endl;
                    if (ELEMENT)
                    {
                        cout << "Now number 4-9 have control to the object" << endl;
                    }
                    else
                    {
                        cout << "Now number 4-9 have control to the curve" << endl;
                    }
                    editMode = TRANSLATION;
                }
                else
                {
                    editMode = FREE;
                }
            }
            break;
        case GLFW_KEY_P:
            if (action == GLFW_RELEASE)
            {
                if (editMode != DELETE)
                {
                    cout << "Switch into delete mode" << endl;
                    editMode = DELETE;
                }
                else
                {
                    editMode = FREE;
                }
            }
            break;
        case GLFW_KEY_1:
            if (action == GLFW_RELEASE)
            {
                if (editMode != INSERT)
                {
                    cout << "Switch into insert mode first before add elements" << endl;
                }
                else
                {
                    vertices.resize(3, 8);
                    index.resize(3, 12);
                    vertices << 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5,
                                0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5,
                                0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5;
                    index << 0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 4, 4,
                            1, 2, 5, 6, 6, 7, 7, 4, 4, 5, 5, 6,
                            2, 3, 6, 2, 7, 3, 4, 0, 5, 1, 6, 7;
                    elements.push_back(new Element(vertices, index));
                    targetElement = elements.size() - 1;
                }
            }
            break;
        case GLFW_KEY_2:
            if (action == GLFW_RELEASE)
            {
                if (editMode != INSERT)
                {
                    cout << "Switch into insert mode first before add elements" << endl;
                }
                else
                {
                    vertices.resize(3, 502);
                    index.resize(3, 1000);
                    ReadOffFile("../data/bumpy_cube.off", vertices, index);
                    elements.push_back(new Element(vertices, index));
                    targetElement = elements.size() - 1;
                }
            }
            break;
        case GLFW_KEY_3:
            if (action == GLFW_RELEASE)
            {
                if (editMode != INSERT)
                {
                    cout << "Switch into insert mode first before add elements" << endl;
                }
                else
                {
                    vertices.resize(3, 502);
                    index.resize(3, 1000);
                    ReadOffFile("../data/bunny.off", vertices, index);
                    elements.push_back(new Element(vertices, index));
                    targetElement = elements.size() - 1;
                }
            }
            break;
        case GLFW_KEY_4:
        case GLFW_KEY_5:
        case GLFW_KEY_6:
        case GLFW_KEY_7:
        case GLFW_KEY_8:
        case GLFW_KEY_9:
            if (action == GLFW_RELEASE)
            {
                if (editMode != TRANSLATION)
                {
                    cout << "Switch into translation mode first before translating elements" << endl;
                }
                else
                {
                    if (targetElement == -1)
                    {
                        cout << "No element selected." << endl;
                    }
                    else
                    {
                        int row = (key - GLFW_KEY_4) / 2;
                        int sign = (key - GLFW_KEY_4) % 2;
                        if (ELEMENT)
                        {
                            Matrix4f tranToward;
                            tranToward << 1, 0, 0, 0,
                                        0, 1, 0, 0,
                                        0, 0, 1, 0,
                                        0, 0, 0, 1;
                            tranToward(row, 3) = (2 * sign - 1) * -0.1;
                            elements[targetElement]->view = tranToward * elements[targetElement]->view;
                        }
                        else
                        {
                            if (elements[targetElement]->bezierCurve.targetCurveVertex == -1)
                            {
                                cout << "No curve vertex in this element" << endl;
                            }
                            else
                            {
                                elements[targetElement]->bezierCurve.V(row, elements[targetElement]->bezierCurve.targetCurveVertex) += (2 * sign - 1) * -0.1;
                                elements[targetElement]->bezierCurve.VBO.update(elements[targetElement]->bezierCurve.V);

                                cout << elements[targetElement]->bezierCurve.V << endl;
                            }
                        }
                    }
                }
            }
            break;
        case GLFW_KEY_T:
            if (action == GLFW_RELEASE)
            {
                if (targetElement == -1)
                {
                    cout << "No element selected." << endl;
                }
                else
                {
                    elements[targetElement]->bezierCurve.V.conservativeResize(elements[targetElement]->bezierCurve.V.rows(), elements[targetElement]->bezierCurve.V.cols() + 1);
                    elements[targetElement]->bezierCurve.V.col(elements[targetElement]->bezierCurve.V.cols() - 1) << 0, 0, 0;
                    elements[targetElement]->bezierCurve.targetCurveVertex = elements[targetElement]->bezierCurve.V.cols() - 1;

                    elements[targetElement]->bezierCurve.VBO.update(elements[targetElement]->bezierCurve.V);
                    cout << elements[targetElement]->bezierCurve.V << endl;
                }
            }
            break;
        case GLFW_KEY_Y:
            if (action == GLFW_RELEASE)
            {
                if (targetElement == -1)
                {
                    cout << "No element selected." << endl;
                }
                else
                {
                    if (elements[targetElement]->bezierCurve.targetCurveVertex == -1)
                    {
                        cout << "No curve vertex in this element" << endl;
                    }
                    else
                    {
                        elements[targetElement]->bezierCurve.targetCurveVertex = (elements[targetElement]->bezierCurve.targetCurveVertex + 1) % (elements[targetElement]->bezierCurve.V.cols());
                    }

                    cout << elements[targetElement]->bezierCurve.targetCurveVertex << endl;
                }
            }
            break;
        case GLFW_KEY_U:
            if (action == GLFW_RELEASE)
            {
                ELEMENT = !ELEMENT;
                if (ELEMENT)
                {
                    cout << "Now number 4-9 have control to the object" << endl;
                }
                else
                {
                    cout << "Now number 4-9 have control to the curve" << endl;
                }
            }
            break;
        case GLFW_KEY_Z:
            if (action == GLFW_RELEASE)
            {
                if (editMode != TRANSLATION)
                {
                    cout << "Switch into translation mode first before translating elements" << endl;
                }
                else
                {
                    if (targetElement == -1)
                    {
                        cout << "No element selected." << endl;
                    }
                    else
                    {
                        cout << "Scale up element by 1.25" << endl;
                        ScaleElement(1.25);
                    }
                }
            }
            break;
        case GLFW_KEY_X:
            if (action == GLFW_RELEASE)
            {
                if (editMode != TRANSLATION)
                {
                    cout << "Switch into translation mode first before translating elements" << endl;
                }
                else
                {
                    if (targetElement == -1)
                    {
                        cout << "No element selected." << endl;
                    }
                    else
                    {
                        cout << "Scale down element by 0.8" << endl;
                        ScaleElement(0.8);
                    }
                }
            }
            break;
        case GLFW_KEY_C:
            if (action == GLFW_RELEASE)
            {
                if (editMode != TRANSLATION)
                {
                    cout << "Switch into translation mode first before translating elements" << endl;
                }
                else
                {
                    if (targetElement == -1)
                    {
                        cout << "No element selected." << endl;
                    }
                    else
                    {
                        cout << "Rotate element by 10 degree counter-clockwise" << endl;
                        RotateElement(10);
                    }
                }
            }
            break;
        case GLFW_KEY_V:
            if (action == GLFW_RELEASE)
            {
                if (editMode != TRANSLATION)
                {
                    cout << "Switch into translation mode first before translating elements" << endl;
                }
                else
                {
                    if (targetElement == -1)
                    {
                        cout << "No element selected." << endl;
                    }
                    else
                    {
                        cout << "Rotate element by 10 degree clockwise" << endl;
                        RotateElement(-10);
                    }
                }
            }
            break;
        case GLFW_KEY_D:
            if (action == GLFW_RELEASE)
            {
                adjustCamera(10, 0);
            }
            break;
        case GLFW_KEY_A:
            if (action == GLFW_RELEASE)
            {
                adjustCamera(-10, 0);
            }
            break;
        case GLFW_KEY_W:
            if (action == GLFW_RELEASE)
            {
                adjustCamera(0, 10);
            }
            break;
        case GLFW_KEY_S:
            if (action == GLFW_RELEASE)
            {
                adjustCamera(0, -10);
            }
            break;
        case GLFW_KEY_R:
            if (action == GLFW_RELEASE)
            {
                radius += 0.5;
                adjustCamera(0, 0);
            }
            break;
        case GLFW_KEY_F:
            if (action == GLFW_RELEASE)
            {
                radius -= 0.5;
                adjustCamera(0, 0);
            }
            break;
        case GLFW_KEY_E:
            if (action == GLFW_RELEASE)
            {
                if (targetElement == -1)
                {
                    cout << "No element selected." << endl;
                }
                else
                {
                    if (!(elements[targetElement]->wireframe))
                    {
                        cout << "Change its shading to flat shading" << endl;
                        elements[targetElement]->wireframe = true;
                        elements[targetElement]->flat = true;
                    }
                    else
                    {
                        if (elements[targetElement]->flat)
                        {
                            cout << "Change its shading to wirefram shading" << endl;
                            elements[targetElement]->flat = false;
                        }
                        else
                        {
                            cout << "Change its shading to phong shading" << endl;
                            elements[targetElement]->wireframe = false;
                        }
                    }
                }
            }
            break;
        case GLFW_KEY_Q:
            if (action == GLFW_RELEASE)
            {
                PERSPECTIVE = !PERSPECTIVE;
            }
            break;
        case GLFW_KEY_B:
            if (action == GLFW_RELEASE)
            {
                cout << "Export SVG" << endl;
                ExportSVG(window);
                cout << "Done" << endl;
            }
            break;
        case GLFW_KEY_SPACE:
            if (action == GLFW_RELEASE)
            {
                ANIMATION = !ANIMATION;
            }
            break;
        case GLFW_KEY_EQUAL:
            if (action == GLFW_RELEASE)
            {
                printf("Increase the zoom by 25%%\n");
                ZoomControl(1.25, 1.25, 1.25, 0, 0, 0);
            }
            break;
        case GLFW_KEY_MINUS:
            if (action == GLFW_RELEASE)
            {
                printf("Decrease the zoom by 25%%\n");
                ZoomControl(0.8, 0.8, 0.8, 0, 0, 0);
            }
            break;
        default:
            break;
    }
}

int main(void)
{
    GLFWwindow* window;

    // Initialize the library
    if (!glfwInit())
        return -1;

    // Activate supersampling
    glfwWindowHint(GLFW_SAMPLES, 8);

    // Ensure that we get at least a 3.2 context
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);

    // On apple we have to load a core profile with forward compatibility
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // Create a windowed mode window and its OpenGL context
    window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

    #ifndef __APPLE__
      glewExperimental = true;
      GLenum err = glewInit();
      if(GLEW_OK != err)
      {
        /* Problem: glewInit failed, something is seriously wrong. */
       fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
      }
      glGetError(); // pull and savely ignonre unhandled errors like GL_INVALID_ENUM
      fprintf(stdout, "Status: Using GLEW %s\n", glewGetString(GLEW_VERSION));
    #endif

    int major, minor, rev;
    major = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MAJOR);
    minor = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MINOR);
    rev = glfwGetWindowAttrib(window, GLFW_CONTEXT_REVISION);
    printf("OpenGL version recieved: %d.%d.%d\n", major, minor, rev);
    printf("Supported OpenGL is %s\n", (const char*)glGetString(GL_VERSION));
    printf("Supported GLSL is %s\n", (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION));

    Sceneview.resize(4, 4);
    Sceneview << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;

    // Initialize the OpenGL Program
    // A program controls the OpenGL pipeline and it must contains
    // at least a vertex shader and a fragment shader to be valid
    Program program;
    const GLchar* vertex_shader =
            "#version 150 core\n"
                    "in vec3 position;"
                    "in vec3 normal;"
                    "uniform int flag;"
                    "out vec3 FragPos;"
                    "out vec3 Normal;"
                    "uniform vec3 normalFace;"
                    "uniform mat4 model;"
                    "uniform mat4 cam;"
                    "uniform mat4 proj;"
                    "uniform mat4 orth;"
                    "uniform mat4 scene;"
                    "void main()"
                    "{"
                    "    FragPos = vec3(model * vec4(position, 1.0));"
                    "    if (flag == 1)"
                    "    {"
                    "       Normal = normalize(mat3(transpose(inverse(model))) * normal);"
                    "    }"
                    "    else"
                    "    {"
                    "       Normal = normalize(mat3(transpose(inverse(model))) * normalFace);"
                    "    }"
                    "    gl_Position = scene * proj * orth * cam * model * vec4(position, 1.0);"
                    "}";
    const GLchar* fragment_shader =
            "#version 150 core\n"
                    "out vec4 outColor;"
                    "in vec3 Normal;"
                    "in vec3 FragPos;"
                    "uniform vec3 triangleColor;"
                    "uniform vec3 lightPos;"
                    "uniform vec3 lightColor;"
                    "uniform vec3 camPos;"
                    "void main()"
                    "{"
                    "    float ambientStrength = 0.1;"
                    "    vec3 ambient = ambientStrength * lightColor;"
                    "    vec3 lightDir = normalize(lightPos - FragPos);"
                    "    float diff = clamp(dot(Normal, lightDir), 0, 1);"
                    "    vec3 diffuse = diff * lightColor;"
                    "    float specularStrength = 0.5;"
                    "    vec3 viewDir = normalize(camPos - FragPos);"
                    "    vec3 reflectDir = reflect(-lightDir, Normal);  "
                    "    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);"
                    "    vec3 specular = specularStrength * spec * lightColor;"
                    "    vec3 result = (ambient + diffuse + specular) * triangleColor;"
                    "    outColor = vec4(result, 1.0);"
                    "}";

    // Compile the two shaders and upload the binary to the GPU
    // Note that we have to explicitly specify that the output "slot" called outColor
    // is the one that we want in the fragment buffer (and thus on screen)
    program.init(vertex_shader, fragment_shader, "outColor");
    program.bind();

    // Register the keyboard callback
    glfwSetKeyCallback(window, key_callback);

    // Register the mouse callback
    glfwSetMouseButtonCallback(window, mouse_button_callback);

    glfwSetCursorPosCallback(window, MousePosCallback);

    // Update viewport
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    
    // Loop until the user closes the window
    while (!glfwWindowShouldClose(window))
    {
        // Bind your program
        program.bind();

        UpdateSceneView(window);

        glUniformMatrix4fv(program.uniform("scene"), 1, GL_FALSE, Sceneview.data());

        // Clear the framebuffer
        glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Enable depth test
        glEnable(GL_DEPTH_TEST);

        if (!ANIMATION)
        {
            for (int i = 0; i < elements.size(); i ++)
            {
                elements[i]->VAO.bind();
                elements[i]->VBO.update(elements[i]->V);
                elements[i]->NBO.update(elements[i]->N);
                elements[i]->EBO.update(elements[i]->E);
                program.bind();
                program.bindVertexAttribArray("position", elements[i]->VBO);
                UpdateCameraMatrix();
                UpdateOrthographicProjection();
                UpdatePerspectiveProjection();

                glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, elements[i]->view.data());
                glUniformMatrix4fv(program.uniform("cam"), 1, GL_FALSE, MCam.data());
                // Upload projection matrix.
                if (PERSPECTIVE)
                {
                    glUniformMatrix4fv(program.uniform("proj"), 1, GL_FALSE, MProj.data());
                    Matrix4f tmp;
                    tmp << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
                    glUniformMatrix4fv(program.uniform("orth"), 1, GL_FALSE, tmp.data());
                }
                else
                {
                    Matrix4f tmp;
                    tmp << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
                    glUniformMatrix4fv(program.uniform("proj"), 1, GL_FALSE, tmp.data());
                    glUniformMatrix4fv(program.uniform("orth"), 1, GL_FALSE, MOrth.data());
                }

                glUniform3f(program.uniform("lightPos"), lightPosition(0), lightPosition(1), lightPosition(2));
                glUniform3f(program.uniform("lightColor"), lightColor(0), lightColor(1), lightColor(2));
                glUniform3f(program.uniform("camPos"), cameraPosition(0), cameraPosition(1), cameraPosition(2));

                int row = elements[i]->E.rows();
                int col = elements[i]->E.cols();

                if (!(elements[i]->wireframe))
                {
                    program.bindVertexAttribArray("normal", elements[i]->NBO);
                    glUniform1i(program.uniform("flag"), 1);
                    if (i == targetElement)
                    {
                        glUniform3f(program.uniform("triangleColor"), elements[i]->selectedColor(0),
                                                                    elements[i]->selectedColor(1), 
                                                                    elements[i]->selectedColor(2));
                    }
                    else
                    {
                        glUniform3f(program.uniform("triangleColor"), elements[i]->color(0),
                                                                    elements[i]->color(1),
                                                                    elements[i]->color(2));
                    }
                    glDrawElements(GL_TRIANGLES, row * col, GL_UNSIGNED_INT, (void*)0);
                }
                else
                {
                    if (elements[i]->flat)
                    {
                        if (i == targetElement)
                        {
                            glUniform3f(program.uniform("triangleColor"), elements[i]->selectedColor(0),
                                                                        elements[i]->selectedColor(1), 
                                                                        elements[i]->selectedColor(2));
                        }
                        else
                        {
                            glUniform3f(program.uniform("triangleColor"), elements[i]->color(0),
                                                                        elements[i]->color(1),
                                                                        elements[i]->color(2));
                        }
                        for (int j = 0; j < col; j ++)
                        {
                            glUniform1i(program.uniform("flag"), 0);
                            glUniform3f(program.uniform("normalFace"), elements[i]->NF(0, j), elements[i]->NF(1, j), elements[i]->NF(2, j));
                            glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, (void*)(sizeof(int) * (3 * j)));
                        }
                    }
                    for (int j = 0; j < col; j ++)
                    {
                        glUniform3f(program.uniform("triangleColor"), 0.0f, 0.0f, 0.0f);
                        glDrawElements(GL_LINE_LOOP, 3, GL_UNSIGNED_INT, (void*)(sizeof(int) * (3 * j)));
                    }
                }

                int controlPointsNum = elements[i]->bezierCurve.V.cols();
                elements[i]->bezierCurve.VAO.bind();
                elements[i]->bezierCurve.VBO.update(elements[i]->bezierCurve.V);
                if (controlPointsNum == 1)
                {
                    program.bindVertexAttribArray("position", elements[i]->bezierCurve.VBO);
                    glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, (elements[i]->bezierCurve.view).data());
                    glPointSize(5.0);
                    glDrawArrays(GL_POINTS, 0, (elements[i]->bezierCurve.V).cols());
                }
                else if (controlPointsNum > 1)
                {
                    program.bindVertexAttribArray("position", elements[i]->bezierCurve.VBO);
                    glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, (elements[i]->bezierCurve.view).data());
                    glPointSize(5.0);
                    glDrawArrays(GL_POINTS, 0, (elements[i]->bezierCurve.V).cols());

                    DrawCruves(&(elements[i]->bezierCurve));
                    elements[i]->bezierCurve.VBO.update(elements[i]->bezierCurve.tmpV);
                    glDrawArrays(GL_LINE_STRIP, 0, 101);
                }
            }
        }

        if (ANIMATION)
        {
            for (int j = 0; j <= 100; j ++)
            {
                glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                for (int i = 0; i < elements.size(); i ++)
                {
                    if ((int)(elements[i]->bezierCurve.V.cols()) < 2)
                    {
                        continue;
                    }
                    elements[i]->VAO.bind();
                    elements[i]->VBO.update(elements[i]->V);
                    elements[i]->NBO.update(elements[i]->N);
                    elements[i]->EBO.update(elements[i]->E);
                    program.bind();
                    program.bindVertexAttribArray("position", elements[i]->VBO);
                    UpdateCameraMatrix();
                    UpdateOrthographicProjection();
                    UpdatePerspectiveProjection();

                    Matrix4f tmp = elements[i]->view;
                    float baryCenterX = elements[i]->baryCenterX, baryCenterY = elements[i]->baryCenterY, baryCenterZ = elements[i]->baryCenterZ;
                    Vector4f world;
                    world << baryCenterX, baryCenterY, baryCenterZ, 1;
                    world = (elements[i]->view) * world;
                    baryCenterX = world(0);
                    baryCenterY = world(1);
                    baryCenterZ = world(2);

                    Matrix4f tranToward;
                    tranToward << 1, 0, 0, -baryCenterX,
                                0, 1, 0, -baryCenterY,
                                0, 0, 1, -baryCenterZ,
                                0, 0, 0, 1;
    
                    Matrix4f tranBack;
                    tranBack << 1, 0, 0, elements[i]->bezierCurve.tmpV(0, j),
                                0, 1, 0, elements[i]->bezierCurve.tmpV(1, j),
                                0, 0, 1, elements[i]->bezierCurve.tmpV(2, j),
                                0, 0, 0, 1;
    
                    tmp = tranBack * tranToward * tmp;
                    glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, tmp.data());
                    glUniformMatrix4fv(program.uniform("cam"), 1, GL_FALSE, MCam.data());
                    // Upload projection matrix.
                    if (PERSPECTIVE)
                    {
                        glUniformMatrix4fv(program.uniform("proj"), 1, GL_FALSE, MProj.data());
                        Matrix4f tmp;
                        tmp << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
                        glUniformMatrix4fv(program.uniform("orth"), 1, GL_FALSE, tmp.data());
                    }
                    else
                    {
                        Matrix4f tmp;
                        tmp << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
                        glUniformMatrix4fv(program.uniform("proj"), 1, GL_FALSE, tmp.data());
                        glUniformMatrix4fv(program.uniform("orth"), 1, GL_FALSE, MOrth.data());
                    }

                    glUniform3f(program.uniform("lightPos"), lightPosition(0), lightPosition(1), lightPosition(2));
                    glUniform3f(program.uniform("lightColor"), lightColor(0), lightColor(1), lightColor(2));
                    glUniform3f(program.uniform("camPos"), cameraPosition(0), cameraPosition(1), cameraPosition(2));

                    int row = elements[i]->E.rows();
                    int col = elements[i]->E.cols();

                    if (!(elements[i]->wireframe))
                    {
                        program.bindVertexAttribArray("normal", elements[i]->NBO);
                        glUniform1i(program.uniform("flag"), 1);
                        if (i == targetElement)
                        {
                            glUniform3f(program.uniform("triangleColor"), elements[i]->selectedColor(0),
                                                                        elements[i]->selectedColor(1), 
                                                                        elements[i]->selectedColor(2));
                        }
                        else
                        {
                            glUniform3f(program.uniform("triangleColor"), elements[i]->color(0),
                                                                        elements[i]->color(1),
                                                                        elements[i]->color(2));
                        }
                        glDrawElements(GL_TRIANGLES, row * col, GL_UNSIGNED_INT, (void*)0);
                    }
                    else
                    {
                        if (elements[i]->flat)
                        {
                            if (i == targetElement)
                            {
                                glUniform3f(program.uniform("triangleColor"), elements[i]->selectedColor(0),
                                                                            elements[i]->selectedColor(1), 
                                                                            elements[i]->selectedColor(2));
                            }
                            else
                            {
                                glUniform3f(program.uniform("triangleColor"), elements[i]->color(0),
                                                                            elements[i]->color(1),
                                                                            elements[i]->color(2));
                            }
                            for (int j = 0; j < col; j ++)
                            {
                                glUniform1i(program.uniform("flag"), 0);
                                glUniform3f(program.uniform("normalFace"), elements[i]->NF(0, j), elements[i]->NF(1, j), elements[i]->NF(2, j));
                                glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, (void*)(sizeof(int) * (3 * j)));
                            }
                        }
                        for (int j = 0; j < col; j ++)
                        {
                            glUniform3f(program.uniform("triangleColor"), 0.0f, 0.0f, 0.0f);
                            glDrawElements(GL_LINE_LOOP, 3, GL_UNSIGNED_INT, (void*)(sizeof(int) * (3 * j)));
                        }
                    }
                }
                // Swap front and back buffers
                glfwSwapBuffers(window);
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            ANIMATION = false;
        }

        // Swap front and back buffers
        glfwSwapBuffers(window);

        // Poll for and process events
        glfwPollEvents();
    }

    // Deallocate opengl memory
    program.free();
    // VAO.free();
    // VBO.free();

    // Deallocate glfw internals
    glfwTerminate();

    return 0;
}