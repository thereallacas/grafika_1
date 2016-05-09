//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2016-tol.
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk.
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kivéve
// - new operatort hivni a lefoglalt adat korrekt felszabaditasa nelkul
// - felesleges programsorokat a beadott programban hagyni
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : yolo
// Neptun : sweg
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
 
#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
 
#include <vector>
 
#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>        // must be downloaded
#include <GL/freeglut.h>    // must be downloaded unless you have an Apple
#endif
 
const unsigned int windowWidth = 600, windowHeight = 600;
 
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Innentol modosithatod...
 
// OpenGL major and minor versions
float PI = 3.14159265359;
float GRAVITY = 200.34;
float FRICTION = -0.045;
 
int majorVersion = 3, minorVersion = 0;
 
void getErrorInfo(unsigned int handle) {
    int logLen;
    glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
    if (logLen > 0) {
        char * log = new char[logLen];
        int written;
        glGetShaderInfoLog(handle, logLen, &written, log);
        printf("Shader log:\n%s", log);
        delete log;
    }
}
 
// check if shader could be compiled
void checkShader(unsigned int shader, char * message) {
    int OK;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
    if (!OK) {
        printf("%s!\n", message);
        getErrorInfo(shader);
    }
}
 
// check if shader could be linked
void checkLinking(unsigned int program) {
    int OK;
    glGetProgramiv(program, GL_LINK_STATUS, &OK);
    if (!OK) {
        printf("Failed to link shader program!\n");
        getErrorInfo(program);
    }
}
 
// vertex shader in GLSL
const char *vertexSource = R"(
#version 140
precision highp float;
 
uniform mat4 MVP;            // Model-View-Projection matrix in row-major format
 
in vec2 vertexPosition;        // variable input from Attrib Array selected by glBindAttribLocation
in vec3 vertexColor;        // variable input from Attrib Array selected by glBindAttribLocation
out vec3 color;                // output attribute
 
void main() {
    color = vertexColor;                                                        // copy color from input to output
    gl_PointSize=30;
    gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * MVP;         // transform to clipping space
}
)";
 
// fragment shader in GLSL
const char *fragmentSource = R"(
#version 140
precision highp float;
 
in vec3 color;                // variable input: interpolated color of vertex shader
out vec4 fragmentColor;        // output that goes to the raster memory as told by glBindFragDataLocation
uniform vec3 unitycolor;
 
void main() {
    fragmentColor = vec4(unitycolor, 1); // extend RGB to RGBA
}
)";
 
// row-major matrix 4x4
struct mat4 {
    float m[4][4];
public:
    mat4() {}
    mat4(float m00, float m01, float m02, float m03,
         float m10, float m11, float m12, float m13,
         float m20, float m21, float m22, float m23,
         float m30, float m31, float m32, float m33) {
        m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
        m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
        m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
        m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
    }
    
    mat4 operator*(const mat4& right) {
        mat4 result;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                result.m[i][j] = 0;
                for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
            }
        }
        return result;
    }
    operator float*() { return &m[0][0]; }
};
 
 
// 3D point in homogeneous coordinates
struct vec4 {
    float v[4];
 
    vec4(float x = 0, float y = 0, float z = 0, float w = 1) {
        v[0] = x; v[1] = y; v[2] = z; v[3] = w;
    }
    vec4 operator*(const mat4& mat) {
        vec4 result;
        for (int j = 0; j < 4; j++) {
            result.v[j] = 0;
            for (int i = 0; i < 4; i++) result.v[j] += v[i] * mat.m[i][j];
        }
        return result;
    }
    vec4 operator*(float a) { return vec4(v[0] * a, v[1] * a, v[2] * a); 
    }
    vec4 operator/(float a) {return vec4(v[0] / a, v[1] / a, v[2] / a); 
    }
    vec4 operator+(const vec4& other) {
        return vec4(v[0] + other.v[0], v[1] + other.v[1], v[2] + other.v[2],v[3]+other.v[3]);
    }
    vec4 operator-(const vec4& other) {
        return vec4(v[0] - other.v[0], v[1] - other.v[1], v[2] - other.v[2], v[3]-other.v[3]);
    }
    vec4 operator*(const vec4& other) {
        return vec4(v[0] * other.v[0], v[1] * other.v[1], v[2] * other.v[2], v[3]*other.v[3]);
    }
    float dot(const vec4& v1, const vec4& v2) {
        return (v1.v[0] * v2.v[0] + v1.v[1] * v2.v[1] + v1.v[2] * v2.v[2]);
    }
    float Length() { return sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); 
    }
};
 
// 2D camera
struct Camera {
    float wCx, wCy;    // center in world coordinates
    float wWx, wWy;    // width and height in world coordinates
public:
    Camera() {
        Animate(0);
    }
    
    mat4 V() { // view matrix: translates the center to the origin
        return mat4(1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    -wCx, -wCy, 0, 1);
    }
    
    mat4 P() { // projection matrix: scales it to be a square of edge length 2
        return mat4(2 / wWx, 0, 0, 0,
                    0, 2 / wWy, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1);
    }
    
    mat4 Vinv() { // inverse view matrix
        return mat4(1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    wCx, wCy, 0, 1);
    }
    
    mat4 Pinv() { // inverse projection matrix
        return mat4(wWx / 2, 0, 0, 0,
                    0, wWy / 2, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1);
    }
    
    void Animate(float t) {
        wCx = 0;
        wCy = 0;
        wWx = 20;
        wWy = 20;
    }
    void Translate(float posx, float posy){
        wCx=posx;
        wCy=posy;
    }
};
 
// 2D camera
Camera camera;
 
// handle of the shader program
unsigned int shaderProgram;
 
 
class Star{
    unsigned int vao;    // vertex array object id
    float sx, sy;        // scaling
    float wTx, wTy;        // translation
    float angle=0;
    
    int multiplicity;
    float mass;
    vec4 velocity;
    
public:
    float colors[3];
    Star(float r, float g, float b,int n, float m){
        colors[0]=r;
        colors[1]=g;
        colors[2]=b;
        multiplicity=n;
        mass=m;
        velocity.v[0]=0;
        velocity.v[1]=0;
        Animate(0);
    }
    
    void Create() {
        glGenVertexArrays(1, &vao);    // create 1 vertex array object
        glBindVertexArray(vao);        // make it active
        
        unsigned int vbo[2];        // vertex buffer objects
        glGenBuffers(2, &vbo[0]);    // Generate 2 vertex buffer objects
        
        // vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
        glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
        
        vec4 p1(-0.2,0);
        vec4 p2(0,0.8);
        vec4 p3(0.2,0);
        
        std::vector<float> vertices;
        int i;
        for(i=0; i<multiplicity; i++){
            float degree = i*2*PI/multiplicity;
            mat4 rotate_matrix(cosf(degree),     sinf(degree),            0,         0,
                               -sinf(degree),    cosf(degree),             0,         0,
                                       0,                0,             0,         0,
                                       0,                0,             0,         1);
                                    
                                    vertices.push_back((p1*rotate_matrix).v[0]);
                                    vertices.push_back((p1*rotate_matrix).v[1]);
                                    vertices.push_back((p2*rotate_matrix).v[0]);
                                    vertices.push_back((p2*rotate_matrix).v[1]);
                                    vertices.push_back((p3*rotate_matrix).v[0]);
                                    vertices.push_back((p3*rotate_matrix).v[1]);
        }
        glBufferData(GL_ARRAY_BUFFER,      // copy to the GPU
                     vertices.size()*sizeof(float),  // number of the vbo in bytes
                     &vertices[0],           // address of the data array on the CPU
                     GL_STATIC_DRAW);       // copy to that part of the memory which is not modified
        // Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
        glEnableVertexAttribArray(0);
        // Data organization of Attribute Array 0
        glVertexAttribPointer(0,            // Attribute Array 0
                              2, GL_FLOAT,  // components/attribute, component type
                              GL_FALSE,        // not in fixed point format, do not normalized
                              0, NULL);     // stride and offset: it is tightly packed
    }
    void Draw() {
        mat4 M(sx*cosf(angle), sx*sinf(angle), 0, 0,
               -sinf(angle)*sy, sy*cosf(angle), 0, 0,
               0, 0, 0, 0,
               wTx, wTy, 0, 1); // model matrix
        
        mat4 MVPTransform = M * camera.V() * camera.P();
        
        // set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
        int location = glGetUniformLocation(shaderProgram, "MVP");
        if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform); // set uniform variable MVP to the MVPTransform
        else printf("uniform MVP cannot be set\n");
        
        int colorlocation = glGetUniformLocation(shaderProgram, "unitycolor");
        glUniform3fv(colorlocation, 1, colors);
        
        glBindVertexArray(vao);    // make the vao and its vbos active playing the role of the data source
        glDrawArrays(GL_TRIANGLES, 0, multiplicity*3*2);    // draw a single triangle with vertices defined in vao
    }
    
    void Translate(float posx, float posy){
        wTx=posx;
        wTy=posy;
    }
    void Scale(float scalarx,float scalary){
        sx*=scalarx;
        sy*=scalary;
    }
    void Rotate(float theta){
        if(angle>2*PI)
            angle=0;
        angle+=theta;
    }
    void Animate(float t) {
        Rotate(0.01f);
        
        sx = .5f+fabs(sinf(5*t));
        sy = .5f+fabs(sinf(5*t));
    
        colors[0]=(2-(velocity.Length()/25))<0 ? 0.2 : (2-(velocity.Length()/25));
        colors[1]=1-(velocity.Length()/25);
        colors[2]=1-(velocity.Length()/25);
    
    }
    vec4 getPosition(){
        vec4 retval(wTx, wTy,0,1);
        return retval;
    }
    vec4 getVelocity(){
        return velocity;
    }
    float getMass(){
        return mass;
    }
    float getDistanceFromCenter(){
        return (getPosition()-vec4(0,0)).Length();
    }
    void addVelocity(vec4 addition){
        velocity=velocity+addition;
    }
    void position_add(vec4 addition){
        wTx+=addition.v[0];
        wTy+=addition.v[1];
    }
    
    float range_x(Star& otherstar){
         return otherstar.getPosition().v[0]-this->getPosition().v[0];
    }
    float range_y(Star& otherstar){
        return otherstar.getPosition().v[1]-this->getPosition().v[1];
    }
    float range_abs(Star& otherstar){
        float retval = sqrt(pow(range_x(otherstar),2)+pow(range_y(otherstar),2));
        return retval;
    }
    float acc_abs(Star& otherstar){
        return (Gravity(otherstar)/mass);
    }
    float acc_x(Star& otherstar){
        return (range_x(otherstar)/range_abs(otherstar))*acc_abs(otherstar);
    }
    float acc_y(Star& otherstar){
        return (range_y(otherstar)/range_abs(otherstar))*acc_abs(otherstar);
    }
    float Gravity(Star& otherstar){
        if (range_abs(otherstar) < 2)
            return 0;
        float retval = (GRAVITY*(otherstar.getMass()*mass)/(range_abs(otherstar)*range_abs(otherstar)));
        return retval;
    }
    vec4 GravityForce(Star& otherstar){
        float fx=(range_x(otherstar)/range_abs(otherstar))*Gravity(otherstar);
        float fy=(range_y(otherstar)/range_abs(otherstar))*Gravity(otherstar);
        return vec4(fx, fy);
    }
    vec4 gravity_acc(Star& otherstar){
        return GravityForce(otherstar)/mass;
    }
    void addFriction(){
        velocity = velocity + velocity*FRICTION;
    }
};
 
class CatmullRom{
    GLuint vao_for_points;
    GLuint vbo_for_points;
    std::vector<float> points;
    vec4 cps[20];           // control points
    vec4 vs[20];
    float  ts[20];          // time values
    float colors[3];
    int nVertices;
public:
    CatmullRom(){
        nVertices=0;
        colors[0]=0;
        colors[1]=1;
        colors[2]=0;
    }
    int getnVertices(){
        return nVertices;
    }
    void Create() {
        //enable and bind for points data
        glGenVertexArrays(1, &vao_for_points);
        glBindVertexArray(vao_for_points);
        
        glGenBuffers(1, &vbo_for_points); // Generate 1 vertex buffer object
        glBindBuffer(GL_ARRAY_BUFFER, vbo_for_points);
        // Enable the vertex attribute arrays
        glEnableVertexAttribArray(0);  // attribute array 0
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), reinterpret_cast<void*>(0)); 
        // attribute array, components/attribute, component type, normalize?, stride, offset
    }
    void AddPoint(float cX, float cY, float t) {
        if (nVertices >= 20) return;
        vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
        cps[nVertices]=wVertex;
        ts[nVertices]=t;
        
        nVertices++;    
        
        // setup vs
        for (int i = 0; i < nVertices; i++) {
            
            int im1 = mymod(i-1);
            int ip1 = mymod(i+1);
            vec4 ri, rim1, rip1;
            float dtprev, dtnext;
            ri = cps[i];
            rim1 = cps[im1];
            rip1 = cps[ip1];
            dtprev = (i == 0) ? .5f : ts[i] - ts[im1];
            dtnext = (i == nVertices - 1) ? .5f : ts[ip1] - ts[i];
            
            vs[i] = ((rip1 - ri)/(dtnext) + (ri - rim1)/(dtprev)) * .9f;
        }
        points= std::vector<float>();
        for (float dt=ts[0]; dt <= ts[nVertices-1]+0.5; dt+=0.002) {
            vec4 p = r(dt);
            points.push_back(p.v[0]);
            points.push_back(p.v[1]);
    
        }
        glBindBuffer(GL_ARRAY_BUFFER, vbo_for_points);
        glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(float), &points[0], GL_DYNAMIC_DRAW);
    }
    vec4 Hermite(vec4 p0, vec4 v0, float t0,
                 vec4 p1, vec4 v1, float t1,
                 float t ) {
        vec4 a0 = p0;
        vec4 a1 = v0;
        vec4 a2 = (((p1-p0)*3)/((t1-t0)*(t1-t0)))-
        ((v1+(v0*2))/(t1-t0));
        vec4 a3 = (((p0-p1)*2)/((t1-t0)*(t1-t0)*(t1-t0)))+
        ((v1+v0)/((t1-t0)*(t1-t0)));
        vec4 ret = (a3*powf((t-t0), 3))+(a2*powf((t-t0),2))+(a1*(t-t0))+a0;
        return ret;
    }
    //http://stackoverflow.com/questions/4003232/how-to-code-a-modulo-operator-in-c-c-obj-c-that-handles-negative-numbers
    int mymod (int a)
    {
        int b = nVertices;
           int ret = a % b;
           if(ret < 0)
         ret+=b;
           return ret;
    }
    vec4 r(float t) {
        float dt=(ts[nVertices-1]+0.5)-ts[0];
        while(t>dt){
            t=t-dt;
        }
        t = t + ts[0];
        for(int i = 0; i < nVertices; i++) {
            int ip1 = mymod(i+1);
            float ttt = (i == nVertices - 1) ? (ts[i] + 0.5f) : ts[ip1];
            if (ts[i] <= t && t <= ttt){
                return Hermite(cps[i], vs[i], ts[i], cps[ip1], vs[ip1], ttt, t);
            }
        }
        //DEFAULT VAL AS XCODE WAS FAILING TO COMPLILE idk lol
       return NULL;
    }
    
    void Draw() {
        if (nVertices > 0) {
            mat4 VPTransform = camera.V() * camera.P();
            
            int location = glGetUniformLocation(shaderProgram, "MVP");
            if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, VPTransform);
            else printf("uniform MVP cannot be set\n");
            
            int colorlocation = glGetUniformLocation(shaderProgram, "unitycolor");
            glUniform3fv(colorlocation, 1, colors);
          
            glBindVertexArray(vao_for_points);
            glDrawArrays(GL_LINE_STRIP, 0, points.size()/2);
        }
    }
};
 
// The virtual world: collection of two objects
CatmullRom catrom;
Star star1(0.53,0,0.66,8, 4);
Star star2(0.2,0,0.9,12, 6);
Star Lucifer(1,1,1,7,10);
 
bool follow = false;
// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    
    // Create objects by setting up their vertex data on the GPU
    star1.Create();
    star2.Create();
    Lucifer.Create();
    catrom.Create();
    star1.Translate(5, 5);
    star2.Translate(-3, -2);
    // Create vertex shader from string
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    if (!vertexShader) {
        printf("Error in vertex shader creation\n");
        exit(1);
    }
    glShaderSource(vertexShader, 1, &vertexSource, NULL);
    glCompileShader(vertexShader);
    checkShader(vertexShader, "Vertex shader error");
    
    // Create fragment shader from string
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    if (!fragmentShader) {
        printf("Error in fragment shader creation\n");
        exit(1);
    }
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
    glCompileShader(fragmentShader);
    checkShader(fragmentShader, "Fragment shader error");
    
    // Attach shaders to a single program
    shaderProgram = glCreateProgram();
    if (!shaderProgram) {
        printf("Error in shader program creation\n");
        exit(1);
    }
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    
    // Connect Attrib Arrays to input variables of the vertex shader
    glBindAttribLocation(shaderProgram, 0, "vertexPosition"); // vertexPosition gets values from Attrib Array 0
    glBindAttribLocation(shaderProgram, 1, "vertexColor");    // vertexColor gets values from Attrib Array 1
    
    // Connect the fragmentColor to the frame buffer memory
    glBindFragDataLocation(shaderProgram, 0, "fragmentColor");    // fragmentColor goes to the frame buffer memory
    
    // program packaging
    glLinkProgram(shaderProgram);
    checkLinking(shaderProgram);
    // make this program run
    glUseProgram(shaderProgram);
}
 
void onExit() {
    glDeleteProgram(shaderProgram);
    printf("exit");
}
 
// Window has become invalid: Redraw
void onDisplay() {
    glClearColor(0, 0, 0, 1);                            // background color
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
   
    Lucifer.Scale(2,2);
    catrom.Draw();
    star1.Draw();
    star2.Draw();
    Lucifer.Draw();
    glutSwapBuffers();                                    // exchange the two buffers
}
 
// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
    if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
    if (key == 'q') exit(0);
    if (key == ' ') follow=!follow;
}
 
// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
    
}
 
// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
    long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
    float sec = time / 1000.0f;             // convert msec to sec
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
        float cX = 2.0f * pX / windowWidth - 1;    // flip y axis
        float cY = 1.0f - 2.0f * pY / windowHeight;
        catrom.AddPoint(cX, cY, sec);
        glutPostRedisplay();     // redraw
    }
}
 
// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}
 
// Idle event indicating that some time elapsed: do animation here
void onIdle() {
    //actual time
    static long last_time = glutGet(GLUT_ELAPSED_TIME);
    long actual_time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
    float diff = actual_time - last_time;
    
    last_time = actual_time;
    float sec = actual_time / 1000.0f;        // convert msec to sec
    
    const float time_step = 5;
    for(int i = 0; i < diff; i += time_step) {
        float dt = fmin(diff-i, time_step) / 1000.0f;
        if (catrom.getnVertices()>1){
            
            star1.addVelocity(star1.gravity_acc(Lucifer)*dt);
            star1.addFriction();
            star1.position_add(star1.getVelocity()*dt);    
    
            star2.addVelocity(star2.gravity_acc(Lucifer)*dt);
            star2.addFriction();
            star2.position_add(star2.getVelocity()*dt);
        }
    }    
    star1.Animate(sec);                        
    star2.Animate(sec);
    Lucifer.Animate(sec);
    
    vec4 pos = catrom.r(sec);
    Lucifer.Translate(pos.v[0], pos.v[1]);
    if (follow){
        camera.Translate(pos.v[0], pos.v[1]);
    }
    glutPostRedisplay();                    // redraw the scene
}
 
// Idaig modosithatod...
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
int main(int argc, char * argv[]) {
    glutInit(&argc, argv);
#if !defined(__APPLE__)
    glutInitContextVersion(majorVersion, minorVersion);
#endif
    glutInitWindowSize(windowWidth, windowHeight);                // Application window is initially of resolution 600x600
    glutInitWindowPosition(100, 100);                            // Relative location of the application window
#if defined(__APPLE__)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_2_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
    glutCreateWindow(argv[0]);
    
#if !defined(__APPLE__)
    glewExperimental = true;    // magic
    glewInit();
#endif
    
    printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
    printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
    printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
    glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
    glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
    printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
    printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
    
    onInitialization();
    
    glutDisplayFunc(onDisplay);                // Register event handlers
    glutMouseFunc(onMouse);
    glutIdleFunc(onIdle);
    glutKeyboardFunc(onKeyboard);
    glutKeyboardUpFunc(onKeyboardUp);
    glutMotionFunc(onMouseMotion);
    
    glutMainLoop();
    onExit();
    return 1;
}