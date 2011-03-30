// CUDA state holder

#ifndef _CUDASTATE_HPP
#define _CUDASTATE_HPP

#include <string>
#include <map>
#include <cuda.h>

using namespace std;

class StateHolder;

class StateEntity
{
  private:
    int count;
    string name;
    StateHolder* owner;

  public:
    StateEntity(string entityName);
    StateEntity(const StateEntity &o);
    
    static int getNewRef();
    string getName();
    int getCount();
    void setName(string new_name);
    void setOwner(StateHolder* new_owner);
    StateHolder* getOwner();

    virtual void allocateGPUMem() = 0;
    virtual void freeGPUMem() = 0;
    virtual void transferHtoD() = 0;
    virtual void transferDtoH();
};

class Quadrature : public StateEntity
{
  private:
    int degree, loc, ngi;
    double *host_weight, *cuda_weight;

  public:
    Quadrature(int degree, int loc, int ngi, string name);
    Quadrature(const Quadrature &o);
    
    int getNumQuadPoints();

    void setHostPointers(double *weight);
    virtual void allocateGPUMem();
    virtual void freeGPUMem();
    virtual void transferHtoD();
    double *getWeights();
};

class Element : public StateEntity
{
  private:
    int dim, loc, ngi;
    double *host_n, *host_dn, *cuda_n, *cuda_dn;
    Quadrature *quad;

  public:
    Element(int dim, int loc, int ngi, string name);
    Element(const Element &o);

    void setHostPointers(double *n, double *dn);
    virtual void allocateGPUMem();
    virtual void freeGPUMem();
    virtual void transferHtoD();

    int getLoc();
    int getDim();
    int getNgi();
    void setQuadrature(Quadrature *new_quad);
    Quadrature* getQuadrature();
    double *getN();
    double *getDn();
};

class CsrSparsity : public StateEntity
{
  private:
    int *host_findrm, *cuda_findrm;
    int *host_colm, *cuda_colm;
    int size_findrm, size_colm;

  public:
    CsrSparsity(int size_findrm, int size_colm, string name);
    CsrSparsity(const CsrSparsity &o);

    void setHostPointers(int *findrm, int *colm);
    int getNumNonZeroes();
    int* getCudaFindrm();
    int* getCudaColm();
    int getSizeFindrm();
    int getSizeColm();
    virtual void allocateGPUMem();
    virtual void freeGPUMem();
    virtual void transferHtoD();
};

class Mesh : public StateEntity
{
  private:
    int *host_ndglno, *cuda_ndglno;
    int dim, num_ele, num_nodes, num_nodes_per_ele;
    Element *shape;

    // Computed by transform_to_physical, and are a property of the mesh.
    double *cuda_transformed_dn;
    double *cuda_detwei;
    
    CsrSparsity *atranspose, *connectivity;

  public:
    Mesh(int num_ele, int num_nodes, int n_dim, string name);
    Mesh(const Mesh &o);

    void setHostPointers(int *ndglno);
    virtual void allocateGPUMem();
    virtual void freeGPUMem();
    virtual void transferHtoD();

    int getNumEle();
    int getNumNodes();
    int getNumNodesPerEle();
    Element* getShape();
    int* getCudaNdglno();
    double* getDetwei();
    double* getCudaTransformedDn();
    void setElement(Element *new_shape);
    void setConnSparsity(CsrSparsity *new_sparsity);
    CsrSparsity* getConnSparsity();
    void setATSparsity(CsrSparsity *new_sparsity);
};

class Field : public StateEntity
{
  protected:
    double *host_val, *cuda_compact_val, *cuda_expanded_val;
    int rank, dim;
    Mesh *mesh;

  public:
    Field(int rank, int dim, string name);
    Field(const Field &o);

    void setHostPointers(double *val);
    virtual void allocateGPUMem();
    virtual void freeGPUMem();
    virtual void transferHtoD() = 0;
    virtual void transferDtoH() = 0;

    virtual int getCompactValSize() const = 0;
    virtual int getExpandedValSize() const = 0;
    void setMesh(Mesh *new_mesh);
    Mesh *getMesh();
    double* getVal();
    int getDim();
    int getRank();
};

class ScalarField : public Field 
{
  public:
    ScalarField(int dim, string name);
    ScalarField(const ScalarField &o);
    
    virtual void transferHtoD();
    virtual void transferDtoH();
    virtual int getCompactValSize() const;
    virtual int getExpandedValSize() const;
};


class VectorField : public Field 
{
  private:
    double *host_val_2, *host_val_3;
    double *cuda_compact_val_2, *cuda_compact_val_3;
    double *cuda_expanded_val_2, *cuda_expanded_val_3;

  public:
    VectorField(int dim, string name);
    VectorField(const VectorField &o);
    
    void setHostPointers(double *val_1, double *val_2, double *val_3);

    virtual void transferHtoD();
    virtual void transferDtoH();
    virtual int getCompactValSize() const;
    virtual int getExpandedValSize() const;
};


class TensorField : public Field 
{
  public:
    TensorField(int dim, string name);
    TensorField(const TensorField &o);
    
    virtual void transferHtoD();
    virtual void transferDtoH();
    virtual int getCompactValSize() const;
    virtual int getExpandedValSize() const;
};

typedef map<int, StateEntity*> EntityMap;
typedef map<string, Field*> FieldMap;
typedef EntityMap::iterator EntityIterator;
typedef FieldMap::iterator FieldIterator;

class StateHolder
{
  private:
    EntityMap entities;
    FieldMap fields;
    CUcontext ctx;

  public:
    StateHolder();
    void initialise();
    void transferAllFields();
    void allocateAllGPUMemory();
    void extractField(string fieldName, int rank);
    void initialiseVector(int id);
    void transferFieldHtoD(string field);
    void transferFieldDtoH(string field);
    void insertEntity(StateEntity *entity);
    void insertField(Field *field);
    void insertTemporaryField(string newFieldName, string likeFieldName);
    // We're assuming that there's only one mesh upon which all function spaces
    // are build for now, so we can get the number of elements and detwei for
    // any field and it will do.
    int getNumEle();
    int getNumNodes();
    int* getEleNodes();
    double* getDetwei();
    double* getCoordinates();
    double* getReferenceDn();
    double* getQuadWeights();
    double* getReferenceN();
    double* getBasisFunction(string fieldName);
    double* getBasisFunctionDerivative(string fieldName);
    double* getElementValue(string fieldName);
    CsrSparsity* getSparsity(string fieldName);
    int getNodesPerEle(string fieldName);
    int getDimension(string fieldName);
    int getValsPerNode(string fieldName);
    int getNumQuadPoints(string fieldName);
    // For returning fields
    void returnFieldToHost(string hostFieldName, string gpuFieldName);
    double* getNodeValue(string fieldName);
    Field *getField(string fieldName);
    ~StateHolder();
};

#endif
