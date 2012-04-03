// CUDA State holder

#include <iostream>
#include <fstream>
#include <cstdio>
#include <cuda.h>
#include "cudastate.hpp"
#include "cudastatic.hpp"

#define grid 128
#define block 128

// Debugging

#define DEBUG_MEM 0

// Memory management

#if DEBUG_MEM
static unsigned long inKB(unsigned long bytes)
{
  return bytes/1024;
}

static unsigned long inMB(unsigned long bytes)
{
  return bytes/(1024*1024);
}


static void printStats(unsigned long free, unsigned long total)
{
  printf("^^^^ Free : %lu bytes (%lu KB) (%lu MB)\n", free, inKB(free), inMB(free));
  printf("^^^^ Total: %lu bytes (%lu KB) (%lu MB)\n", total, inKB(total), inMB(total));
  printf("^^^^ %f%% free, %f%% used\n", 100.0*free/(double)total, 100.0*(total - free)/(double)total);
}
#endif

void check(cudaError_t result)
{
  if(result!=0) {
    cerr << "CUDA Error: " << cudaGetErrorString(result) << endl;
    exit(result);
  }
}

void allocate(void **ptr, int length)
{

  #if DEBUG_MEM
    cout << "Allocating " << length << " bytes" << endl;
  #endif

  cudaError_t result;

  result = cudaMalloc(ptr, length);
  check(result);

  #if DEBUG_MEM
  CUresult res;
  size_t free, total;

  res = cuMemGetInfo(&free, &total);

  if(res!=CUDA_SUCCESS)
  {
    cerr << "Driver API Error" << endl;
  }

    printStats(free, total);
  #endif
}

void deallocate(void *ptr)
{
  cudaError_t result;

  result = cudaFree(ptr);
  check(result);
}

void copyHtoD(void *host_ptr, void *dev_ptr, int length)
{
  cudaError_t result;

  #if DEBUG_MEM
    cout << "Copy " << host_ptr << "(host) to " << dev_ptr << "(dev), len " << length << endl;
  #endif

  result = cudaMemcpy(dev_ptr, host_ptr, length, cudaMemcpyHostToDevice);
  check(result);
}

void copyDtoH(void *dev_ptr, void *host_ptr, int length)
{
  cudaError_t result;

  #if DEBUG_MEM
    cout << "Copy " << dev_ptr << "(dev) to " << host_ptr << "(host), len " << length << endl;
  #endif

  result = cudaMemcpy(host_ptr, dev_ptr, length, cudaMemcpyDeviceToHost);
  check(result);
}

// StateEntity
// The base of all entities in state - most methods are
// empty as we don't need them to do anything, but
// would like to visit every entity easily.

static int counter = 0;

StateEntity::StateEntity(string entityName)
{
  name = entityName;
  count = getNewRef();
}

StateEntity::StateEntity(const StateEntity &o)
{
  name = o.name;
  owner = o.owner;
  count = getNewRef();
  owner->insertEntity(this);
}

int StateEntity::getNewRef()
{
  int ref = counter;
  counter++;
  return ref;
}

string StateEntity::getName()
{
  return name;
}

int StateEntity::getCount()
{
  return count;
}

void StateEntity::setName(string new_name)
{
  name = new_name;
}

void StateEntity::setOwner(StateHolder* new_owner)
{
  owner = new_owner;
}

StateHolder* StateEntity::getOwner()
{
  return owner;
}

void StateEntity::transferDtoH()   { }

// Quadrature

Quadrature::Quadrature(int degree, int loc, int ngi, string name="Anon_quad") : StateEntity(name), degree(degree), loc(loc), ngi(ngi) { }

Quadrature::Quadrature(const Quadrature &o) : StateEntity(o)
{
  degree = o.degree;
  loc = o.loc;
  ngi = o.ngi;

  host_weight = o.host_weight;
  allocate((void**)&cuda_weight, sizeof(double)*ngi);
  cudaMemcpy(cuda_weight, o.cuda_weight, sizeof(double)*ngi, cudaMemcpyDeviceToDevice);
}

int Quadrature::getNumQuadPoints()
{
  return ngi;
}

void Quadrature::setHostPointers(double *weight)
{
  host_weight = weight;
}

void Quadrature::allocateGPUMem()
{
  allocate((void**)&cuda_weight, sizeof(double)*ngi);
}

void Quadrature::freeGPUMem()
{
  deallocate(cuda_weight);
}

void Quadrature::transferHtoD()
{
  copyHtoD(host_weight, cuda_weight, sizeof(double)*ngi);
}

double* Quadrature::getWeights()
{
  return cuda_weight;
}

// Element

Element::Element(int dim, int loc, int ngi, string name="Anon_element") : StateEntity(name), dim(dim), loc(loc), ngi(ngi) { }

Element::Element(const Element &o) : StateEntity(o)
{
  dim = o.dim;
  loc = o.loc;
  ngi = o.ngi;

  allocate((void**)&cuda_n, sizeof(double)*loc*ngi);
  allocate((void**)&cuda_dn, sizeof(double)*loc*ngi*dim);
  cudaMemcpy(cuda_n, o.cuda_n, sizeof(double)*loc*ngi, cudaMemcpyDeviceToDevice);
  cudaMemcpy(cuda_dn, o.cuda_dn, sizeof(double)*loc*ngi*dim, cudaMemcpyDeviceToDevice);

  quad = new Quadrature(*(o.quad));
}

void Element::setHostPointers(double *n, double *dn)
{
  host_n = n;
  host_dn = dn;
}

void Element::allocateGPUMem()
{
  allocate((void**)&cuda_n, sizeof(double)*loc*ngi);
  allocate((void**)&cuda_dn, sizeof(double)*loc*ngi*dim);
}

void Element::freeGPUMem()
{
  deallocate(cuda_n);
  deallocate(cuda_dn);
}

void Element::transferHtoD()
{
  quad->transferHtoD();
  copyHtoD(host_n, cuda_n, sizeof(double)*loc*ngi);
  copyHtoD(host_dn, cuda_dn, sizeof(double)*loc*ngi*dim);
}

int Element::getLoc()
{
  return loc;
}

int Element::getDim()
{
  return dim;
}

int Element::getNgi()
{
  return ngi;
}

void Element::setQuadrature(Quadrature *new_quad)
{
  quad = new_quad;
}

Quadrature* Element::getQuadrature()
{
  return quad;
}

double* Element::getN()
{
  return cuda_n;
}

double* Element::getDn()
{
  return cuda_dn;
}

// CsrSparsity

CsrSparsity::CsrSparsity(int size_findrm, int size_colm, string name="Anon_sparsity") : StateEntity(name), size_findrm(size_findrm), size_colm(size_colm) { }

CsrSparsity::CsrSparsity(const CsrSparsity &o) : StateEntity(o)
{
  size_findrm = o.size_findrm;
  size_colm = o.size_colm;

  allocate((void**)&cuda_findrm, sizeof(int)*size_findrm);
  allocate((void**)&cuda_colm, sizeof(int)*size_colm);
  cudaMemcpy(cuda_findrm, o.cuda_findrm, sizeof(int)*size_findrm, cudaMemcpyDeviceToDevice);
  cudaMemcpy(cuda_colm, o.cuda_colm, sizeof(int)*size_colm, cudaMemcpyDeviceToDevice);
}

void CsrSparsity::setHostPointers(int *findrm, int *colm)
{
  host_findrm = findrm;
  host_colm = colm;
}

int CsrSparsity::getNumNonZeroes()
{
  return size_colm;
}

int* CsrSparsity::getCudaFindrm()
{
  return cuda_findrm;
}

int* CsrSparsity::getCudaColm()
{
  return cuda_colm;
}

int CsrSparsity::getSizeFindrm()
{
  return size_findrm;
}

int CsrSparsity::getSizeColm()
{
  return size_colm;
}

void CsrSparsity::allocateGPUMem()
{
  allocate((void**)&cuda_findrm, sizeof(int)*size_findrm);
  allocate((void**)&cuda_colm, sizeof(int)*size_colm);
}

void CsrSparsity::freeGPUMem()
{
  deallocate(cuda_findrm);
  deallocate(cuda_colm);
}

void CsrSparsity::transferHtoD()
{
  copyHtoD(host_findrm, cuda_findrm, sizeof(int)*size_findrm);
  copyHtoD(host_colm, cuda_colm, sizeof(int)*size_colm);
}

// Mesh

Mesh::Mesh(int num_ele, int num_nodes, int n_dim, string name="Anon_mesh") : StateEntity(name), num_ele(num_ele), num_nodes(num_nodes), dim(n_dim) { }

Mesh::Mesh(const Mesh &o) : StateEntity(o)
{
  num_ele = o.num_ele;
  num_nodes = o.num_nodes;
  dim = o.dim;

  shape = new Element(*(o.shape));
  int num_local_nodes = num_ele * shape->getLoc();
  allocate((void**)&cuda_ndglno, sizeof(int)*num_local_nodes);
  cudaMemcpy(cuda_ndglno, o.cuda_ndglno, sizeof(int)*num_local_nodes, cudaMemcpyDeviceToDevice);

  connectivity = new CsrSparsity(*(o.connectivity));
  atranspose = new CsrSparsity(*(o.atranspose));

  int transformed_dn_size = shape->getLoc() * num_ele * shape->getNgi() * dim;

  allocate((void**)&cuda_transformed_dn, transformed_dn_size);
  allocate((void**)&cuda_detwei, sizeof(double)*num_ele*shape->getNgi());
  cudaMemcpy(cuda_transformed_dn, o.cuda_transformed_dn, transformed_dn_size, cudaMemcpyDeviceToDevice);
  cudaMemcpy(cuda_detwei, o.cuda_detwei, sizeof(double)*num_ele*shape->getNgi(), cudaMemcpyDeviceToDevice);
}

void Mesh::setHostPointers(int *ndglno)
{
  host_ndglno = ndglno;
}

void Mesh::allocateGPUMem()
{
  int num_local_nodes = num_ele * shape->getLoc();

  allocate((void**)&cuda_ndglno, sizeof(int)*num_local_nodes);

  int transformed_dn_size = (shape->getLoc() * num_ele * shape->getNgi() * dim)*sizeof(double);

  allocate((void**)&cuda_transformed_dn, transformed_dn_size);
  allocate((void**)&cuda_detwei, sizeof(double)*num_ele* shape->getNgi());

}

void Mesh::freeGPUMem()
{
  deallocate(cuda_ndglno);
}

void Mesh::transferHtoD()
{
  int num_local_nodes = num_ele * shape->getLoc();

  shape->transferHtoD();
  atranspose->transferHtoD();
  connectivity->transferHtoD();
  copyHtoD(host_ndglno, cuda_ndglno, sizeof(int)*num_local_nodes);
}

int Mesh::getNumEle()
{
  return num_ele;
}

int Mesh::getNumNodes()
{
  return num_nodes;
}

int Mesh::getNumNodesPerEle()
{
  return shape->getLoc();
}

Element* Mesh::getShape()
{
  return shape;
}

int* Mesh::getCudaNdglno()
{
  return cuda_ndglno;
}

double* Mesh::getDetwei()
{
  return cuda_detwei;
}

double* Mesh::getCudaTransformedDn()
{
  return cuda_transformed_dn;
}

void Mesh::setElement(Element* new_shape)
{
  shape = new_shape;
}

void Mesh::setConnSparsity(CsrSparsity *new_sparsity)
{
  connectivity = new_sparsity;
}

CsrSparsity* Mesh::getConnSparsity()
{
  return connectivity;
}

void Mesh::setATSparsity(CsrSparsity *new_sparsity)
{
  atranspose = new_sparsity;
}

// Field

Field::Field(int rank, int dim, string name="Anon_field") : StateEntity(name), rank(rank), dim(dim) { }

Field::Field(const Field &o) : StateEntity(o)
{
  rank = o.rank;
  dim = o.dim;
  host_val = o.host_val;

  mesh = new Mesh(*(o.mesh));

  allocate((void**)&cuda_compact_val, sizeof(double)*o.getCompactValSize());
  allocate((void**)&cuda_expanded_val, sizeof(double)*o.getExpandedValSize());
  cudaMemcpy(cuda_compact_val, o.cuda_compact_val, sizeof(double)*o.getCompactValSize(), cudaMemcpyDeviceToDevice);
  cudaMemcpy(cuda_expanded_val, o.cuda_expanded_val, sizeof(double)*o.getExpandedValSize(), cudaMemcpyDeviceToDevice);
}

void Field::setHostPointers(double *val)
{
  host_val = val;
}

void Field::allocateGPUMem()
{
  allocate((void**)&cuda_compact_val, sizeof(double)*getCompactValSize());
  allocate((void**)&cuda_expanded_val, sizeof(double)*getExpandedValSize());
}

void Field::freeGPUMem()
{
  deallocate(cuda_compact_val);
  deallocate(cuda_expanded_val);
}

void Field::setMesh(Mesh *new_mesh)
{
  mesh = new_mesh;
}

Mesh* Field::getMesh()
{
  return mesh;
}

double* Field::getCompactVal()
{
  return cuda_compact_val;
}

double* Field::getVal()
{
  return cuda_expanded_val;
}

int Field::getDim()
{
  return dim;
}

int Field::getRank()
{
  return rank;
}

// Scalar_field

ScalarField::ScalarField(int dim, string name="Anon_scalar_field") : Field(0, dim, name)  { }

void ScalarField::transferHtoD()
{
  cout << "Copying scalar field " << getName() << " to GPU." << endl;

  mesh->transferHtoD();
  copyHtoD(host_val, cuda_compact_val, sizeof(double)*getCompactValSize());
  cout << "Expanding data for field " << getName() << "." << endl;
  // The 1 only makes sense for a scalar. Dim would be for a vector, and dim*dim for a tensor.
  expand_data<<<grid,block>>>(cuda_expanded_val, cuda_compact_val, mesh->getCudaNdglno(), mesh->getNumEle(), 1, mesh->getNumNodesPerEle());
  check(cudaGetLastError());
}

void ScalarField::transferDtoH()
{
  contract_data<<<grid,block>>>(cuda_compact_val, cuda_expanded_val, mesh->getCudaNdglno(), mesh->getNumEle(), 1, mesh->getNumNodesPerEle());
  check(cudaGetLastError());
  copyDtoH(cuda_compact_val, host_val, sizeof(double)*getCompactValSize());
}

ScalarField::ScalarField(const ScalarField &o) : Field(o) { }

int ScalarField::getCompactValSize() const
{
  return mesh->getNumNodes();
}

int ScalarField::getExpandedValSize() const
{
  return mesh->getNumEle() * mesh->getShape()->getLoc();
}

// Vector_field

VectorField::VectorField(int dim, string name="Anon_vector_field") : Field(1, dim, name)  { }

VectorField::VectorField(const VectorField &o) : Field(o) { }

void VectorField::transferHtoD()
{
  cout << "Copying vector field " << getName() << " to GPU." << endl;
  mesh->transferHtoD();
  copyHtoD(host_val, cuda_compact_val, sizeof(double)*getCompactValSize());
  expand_data<<<grid,block>>>(cuda_expanded_val, cuda_compact_val, mesh->getCudaNdglno(), mesh->getNumEle(), mesh->getShape()->getDim(), mesh->getNumNodesPerEle());
  check(cudaGetLastError());
}

void VectorField::transferDtoH()
{
  cerr << "Oh no, I haven't implemented this yet. " << __FUNCTION__ << endl;
  exit(-1);
}

int VectorField::getCompactValSize() const
{
  return mesh->getNumNodes() * mesh->getShape()->getDim();
}

int VectorField::getExpandedValSize() const
{
  return mesh->getNumEle() * mesh->getShape()->getLoc() * mesh->getShape()->getDim();
}

// Tensor_field

TensorField::TensorField(int dim, string name="Anon_tensor_field") : Field(2, dim, name)  { }

void TensorField::transferHtoD()
{
  cerr << "Oh no, I haven't implemented this yet." << __FUNCTION__ << endl;
  exit(-1);

  cout << "Copying tensor field " << getName() << " to GPU." << endl;
  mesh->transferHtoD();
  copyHtoD(host_val, cuda_compact_val, sizeof(double)*getCompactValSize());
  int n_vals_per_ele = mesh->getShape()->getDim();
  n_vals_per_ele = n_vals_per_ele*n_vals_per_ele;
  expand_data<<<grid,block>>>(cuda_expanded_val, cuda_compact_val, mesh->getCudaNdglno(), mesh->getNumEle(), n_vals_per_ele, mesh->getNumNodesPerEle());
  check(cudaGetLastError());
}

void TensorField::transferDtoH()
{
  cerr << "Oh no, I haven't implemented this yet." << __FUNCTION__ << endl;
  exit(-1);
}

TensorField::TensorField(const TensorField &o) : Field(o) { }

int TensorField::getCompactValSize() const
{
  int dim = mesh->getShape()->getDim();

  return mesh->getNumNodes() * dim * dim;
}

int TensorField::getExpandedValSize() const
{
  int dim = mesh->getShape()->getDim();

  return mesh->getNumEle() * mesh->getShape()->getLoc() * dim * dim;
}

// StateHolder

StateHolder::StateHolder()
{
  #if DEBUG_MEM
  cout << "Constructing state" << endl;
  #endif
}

StateHolder::~StateHolder()
{
  #if DEBUG_MEM
  cout << "Destroying state" <<endl;
  #endif
  for(EntityIterator i=entities.begin(); i!=entities.end(); ++i)
  {
    #if DEBUG_MEM
    cout << "Deleting object " << i->first << " (" << i->second->getName() << ")" << endl;
    #endif
    i->second->freeGPUMem();
    delete i->second;
  }

  #if DEBUG_MEM
  cuCtxDetach(ctx);
  #endif
}

void StateHolder::initialise()
{
  #if DEBUG_MEM
  cout << "Initialising state..." << endl;
  #endif
  extractField("Coordinate", 1);

  #if DEBUG_MEM
  CUdevice dev;

  cuInit(0);
  cuDeviceGet(&dev,0);
  cuCtxCreate(&ctx, 0, dev);
  #endif
}

void StateHolder::transferAllFields()
{
  cout << "Transferring all fields to GPU" << endl;
  for(FieldIterator i=fields.begin(); i!=fields.end(); ++i)
    i->second->transferHtoD();
}

void StateHolder::allocateAllGPUMemory()
{
  for(EntityIterator i=entities.begin(); i!=entities.end(); ++i)
    i->second->allocateGPUMem();
}

void StateHolder::transferFieldHtoD(string field)
{
  cout << "Transferring field " << field << " to GPU" << endl;
  fields[field]->transferHtoD();
}

void StateHolder::transferFieldDtoH(string field)
{
  cout << "Transferring field " << field << " to host" << endl;
  fields[field]->transferDtoH();
}

void StateHolder::insertEntity(StateEntity *entity)
{
  entity->setOwner(this);
  entities[entity->getCount()] = entity;
}

void StateHolder::insertField(Field *field)
{
  fields[field->getName()] = field;
}

void StateHolder::insertTemporaryField(string newFieldName, string likeFieldName)
{
  Field *f = fields[likeFieldName];
  ScalarField *sf = dynamic_cast<ScalarField*>(f);
  VectorField *vf = dynamic_cast<VectorField*>(f);
  TensorField *tf = dynamic_cast<TensorField*>(f);
  Field *newField;

  if (sf)
  {
    newField = new ScalarField(*sf);
  }
  else if (vf)
  {
    newField = new VectorField(*vf);
  }
  else if (tf)
  {
    newField = new TensorField(*tf);
  }
  else
  {
    cerr << "Copied field " << likeFieldName << " not found when trying to make new field " << newFieldName << endl;
  }

  newField->setName(newFieldName);
  insertField(newField);
}

// Prototype for accessing fortran function. The prototype can't go in the
// extractField method due to the linkage specification.
extern "C" void extract_scalar_field_wrapper(const char*, int*, int**, int*, int**, int*,
		    		             int**, int*, int**, int*,
					     int*, int*, int**, int*, int*,
					     int*, double**, double**, int*, double**, double**);

extern "C" void extract_vector_field_wrapper(const char*, int*, int**, int*, int**, int*,
		    		             int**, int*, int**, int*,
					     int*, int*, int**, int*, int*,
					     int*, double**, double**, int*, double**, double**);

void StateHolder::extractField(string field_name, int rank)
{
  const char *fortran_name = field_name.c_str();
  int fortran_length = field_name.length();

  int *at_findrm, at_findrm_size, *at_colm, at_colm_size, *conn_findrm, conn_findrm_size, *conn_colm, conn_colm_size;
  int num_ele, num_nodes, *ndglno, loc, dim, ngi, degree;
  double *n, *dn, *weight, *val;

  // Call to fortran function
  switch (rank)
  {
    case 0: // scalar field
      extract_scalar_field_wrapper(fortran_name, &fortran_length, &conn_findrm, &conn_findrm_size, &conn_colm, &conn_colm_size,
                                      &at_findrm, &at_findrm_size, &at_colm, &at_colm_size,
  				      &num_ele, &num_nodes, &ndglno, &loc, &dim,
				      &ngi, &n, &dn, &degree, &weight, &val);
      break;

    case 1: // vector field
      extract_vector_field_wrapper(fortran_name, &fortran_length, &conn_findrm, &conn_findrm_size, &conn_colm, &conn_colm_size,
                                      &at_findrm, &at_findrm_size, &at_colm, &at_colm_size,
  				      &num_ele, &num_nodes, &ndglno, &loc, &dim,
				      &ngi, &n, &dn, &degree, &weight, &val);
      break;

    case 2: // tensor field
      cerr << "Error: tensor field extraction not implemented." << endl;
      break;
  }

  Quadrature *quad = new Quadrature(degree, loc, ngi, field_name+"Quadrature");
  quad->setHostPointers(weight);
  insertEntity(quad);

  Element *shape = new Element(dim, loc, ngi, field_name+"Element");
  shape->setHostPointers(n, dn);
  shape->setQuadrature(quad);
  insertEntity(shape);

  CsrSparsity *conn_sparsity = new CsrSparsity(conn_findrm_size, conn_colm_size, field_name+"ConnSparsity");
  conn_sparsity->setHostPointers(conn_findrm, conn_colm);
  insertEntity(conn_sparsity);

  CsrSparsity *at_sparsity = new CsrSparsity(at_findrm_size, at_colm_size, field_name+"ATSparsity");
  at_sparsity->setHostPointers(at_findrm, at_colm);
  insertEntity(at_sparsity);

  Mesh *mesh = new Mesh(num_ele, num_nodes, dim, field_name+"Mesh");
  mesh->setHostPointers(ndglno);
  mesh->setConnSparsity(conn_sparsity);
  mesh->setATSparsity(at_sparsity);
  mesh->setElement(shape);
  insertEntity(mesh);

  Field *field;

  switch (rank)
  {
    case 0:
      field = new ScalarField(dim, field_name);
      break;

    case 1:
      field = new VectorField(dim, field_name);
      break;

    case 2:
      field = new TensorField(dim, field_name);
      break;
  }

  field->setHostPointers(val);

  field->setMesh(mesh);
  insertEntity(field);
  insertField(field);
}

double* StateHolder::getCoordinates()
{
  return fields["Coordinate"]->getVal();
}

int StateHolder::getNumEle(string fieldName)
{
  return fields[fieldName]->getMesh()->getNumEle();
}

int StateHolder::getNumNodes(string fieldName)
{
  return fields[fieldName]->getMesh()->getNumNodes();
}

double* StateHolder::getDetwei(string fieldName)
{
  return fields[fieldName]->getMesh()->getDetwei();
}

int* StateHolder::getEleNodes(string fieldName)
{
  return fields[fieldName]->getMesh()->getCudaNdglno();
}

double* StateHolder::getReferenceDn(string fieldName)
{
  return fields[fieldName]->getMesh()->getShape()->getDn();
}

double* StateHolder::getQuadWeights(string fieldName)
{
  return fields[fieldName]->getMesh()->getShape()->getQuadrature()->getWeights();
}

double* StateHolder::getReferenceN(string fieldName)
{
  return fields[fieldName]->getMesh()->getShape()->getN();
}

int StateHolder::getDimension(string fieldName)
{
  return fields[fieldName]->getDim();
}

int StateHolder::getValsPerNode(string fieldName)
{
  Field* f = fields[fieldName];
  return pow(f->getDim(),  f->getRank());
}

int StateHolder::getNodesPerEle(string fieldName)
{
  return fields[fieldName]->getMesh()->getNumNodesPerEle();
}

int StateHolder::getNumQuadPoints(string fieldName)
{
  return fields[fieldName]->getMesh()->getShape()->getQuadrature()->getNumQuadPoints();
}

double* StateHolder::getBasisFunction(string fieldName)
{
  return fields[fieldName]->getMesh()->getShape()->getN();
}

double* StateHolder::getBasisFunctionDerivative(string fieldName)
{
  return fields[fieldName]->getMesh()->getCudaTransformedDn();
}

double* StateHolder::getElementValue(string fieldName)
{
  return fields[fieldName]->getVal();
}

CsrSparsity* StateHolder::getSparsity(string fieldName)
{
  return fields[fieldName]->getMesh()->getConnSparsity();
}

Field* StateHolder::getField(string fieldName)
{
  return fields[fieldName];
}

void StateHolder::returnFieldToHost(string fieldName)
{
  fields[fieldName]->transferDtoH();
}

void StateHolder::returnFieldToHost(string targetFieldName, string sourceFieldName)
{
  Field* targetField = fields[targetFieldName];
  Field* sourceField = fields[sourceFieldName];
  int len = sizeof(double) * sourceField->getExpandedValSize();
  cudaMemcpy(targetField->getVal(), sourceField->getVal(), len, cudaMemcpyDeviceToDevice);
  sourceField->transferDtoH();
}

void matrix_dump(int* findrm, int* colm, double* val, int findrm_size, int colm_size, const char* filename)
{
  int *host_findrm = new int[findrm_size];
  int *host_colm = new int[colm_size];
  double *host_val = new double[colm_size];

  copyDtoH(findrm, host_findrm, findrm_size*sizeof(int));
  copyDtoH(colm, host_colm, colm_size*sizeof(int));
  copyDtoH(val, host_val, colm_size*sizeof(double));

  ofstream f(filename);
  f.precision(16);
  f.setf(ios::scientific);
  // Matrix Market header
  f << "%%MatrixMarket matrix coordinate real general\n";
  // rows cols nonzeros
  f << findrm_size-1 << " " << findrm_size-1 << " " << colm_size << "\n";
  // row col value (row, col are 1-based)
  for (int row = 0; row < findrm_size-1; row++) {
    /*cout << "row " << row << endl;*/
    for (int col = host_findrm[row]-1; col < host_findrm[row+1]-1; col++) {
      f << row+1 << " " << host_colm[col] << " " << host_val[col] << "\n";
    }
  }
  f.close();

  delete [] host_findrm;
  delete [] host_colm;
  delete [] host_val;
}

void vector_dump(double* val, int size, const char* filename)
{
  double *host_val = new double[size];
  copyDtoH(val, host_val, size*sizeof(double));
  ofstream f(filename);
  f.precision(16);
  f.setf(ios::scientific);
  for (int i = 0; i < size; i++) {
    f << host_val[i] << "\n";
  }
  f.close();
  delete [] host_val;
}

void host_vector_dump(double* val, int size, const char* filename)
{
  ofstream f(filename);
  f.precision(16);
  f.setf(ios::scientific);
  for (int i = 0; i < size; i++) {
    f << val[i] << "\n";
  }
  f.close();
}
