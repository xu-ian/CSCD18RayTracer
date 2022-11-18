/*
   utils.c - F.J. Estrada, Dec. 9, 2010

   Utilities for the ray tracer. You will need to complete
   some of the functions in this file. Look for the sections
   marked "TO DO". Be sure to read the rest of the file and
   understand how the entire code works.

   HOWEVER: Note that there are a lot of incomplete functions
            that will only be used for the advanced ray tracer!
	    So, read the handout carefully and implement only
	    the code you need for the corresponding assignment.
   
   Last updated: Aug. 2017  -  F.J.E.
*/

/*****************************************************************************
* COMPLETE THIS TEXT BOX:
*
* 1) Student Name:	Ian Xu	
* 2) Student Name:		
*
* 1) Student number: 1006319208
* 2) Student number:
* 
* 1) UtorID: xuian
* 2) UtorID:
* 
* We hereby certify that the work contained here is our own
*
* ________Ian_Xu______             _____________________
* (sign with your name)            (sign with your name)
********************************************************************************/

#include "utils.h"

// A useful 4x4 identity matrix which can be used at any point to
// initialize or reset object transformations
double eye4x4[4][4]={{1.0, 0.0, 0.0, 0.0},
                    {0.0, 1.0, 0.0, 0.0},
                    {0.0, 0.0, 1.0, 0.0},
                    {0.0, 0.0, 0.0, 1.0}};

/////////////////////////////////////////////
// Primitive data structure section
/////////////////////////////////////////////
struct point3D *newPoint(double px, double py, double pz)
{
 // Allocate a new point structure, initialize it to
 // the specified coordinates, and return a pointer
 // to it.

 struct point3D *pt=(struct point3D *)calloc(1,sizeof(struct point3D));
 if (!pt) fprintf(stderr,"Out of memory allocating point structure!\n");
 else
 {
  pt->px=px;
  pt->py=py;
  pt->pz=pz;
  pt->pw=1.0;
 }
 return(pt);
}

struct pointLS *newPLS(struct point3D *p0, double r, double g, double b)
{
 // Allocate a new point light sourse structure. Initialize the light
 // source to the specified RGB colour
 // Note that this is a point light source in that it is a single point
 // in space, if you also want a uniform direction for light over the
 // scene (a so-called directional light) you need to place the
 // light source really far away.

 struct pointLS *ls=(struct pointLS *)calloc(1,sizeof(struct pointLS));
 if (!ls) fprintf(stderr,"Out of memory allocating light source!\n");
 else
 {
  memcpy(&ls->p0,p0,sizeof(struct point3D));	// Copy light source location

  ls->col.R=r;					// Store light source colour and
  ls->col.G=g;					// intensity
  ls->col.B=b;
  ls->isObj = 0;				//Set the light source to point light source
 }
 return(ls);
}

struct pointLS *newALS(struct object3D *obj){
	struct pointLS *as=(struct pointLS *)calloc(1,sizeof(struct pointLS));
	if(!as) fprintf(stderr,"Out of memory allocating light source!\n");
	else
	{
		memcpy(&as->obj,obj,sizeof(struct object3D));	// Copy light source object
		as->isObj = 1;			//Set the light source to area light source
	}
	return(as);
}


//Helper function to compute the value of a ray at a given lambda
void computePoint(struct ray3D *r, double lambda, struct point3D *output){
	struct point3D value = r->d;
	scalarMultVector(&value, lambda);
	addVectors(&r->p0, &value);
	*output = value;
}

/////////////////////////////////////////////
// Ray and normal transforms
/////////////////////////////////////////////
inline void rayTransform(struct ray3D *ray_orig, struct ray3D *ray_transformed, struct object3D *obj)
{
 // Transforms a ray using the inverse transform for the specified object. This is so that we can
 // use the intersection test for the canonical object. Note that this has to be done carefully!

 ///////////////////////////////////////////
 // TO DO: Complete this function
 ///////////////////////////////////////////
 
 //Creates a copy so the original ray does not get changed
 *ray_transformed = *ray_orig;
 
  
 //Transforms the point and the direction based on the inverse matrix
 matVecMult(obj->Tinv, &ray_transformed->d);
 matVecMult(obj->Tinv, &ray_transformed->p0);
}

inline void normalTransform(struct point3D *n_orig, struct point3D *n_transformed, struct object3D *obj)
{
 // Computes the normal at an affinely transformed point given the original normal and the
 // object's inverse transformation. From the notes:
 // n_transformed=A^-T*n normalized.

 ///////////////////////////////////////////
 // TO DO: Complete this function
 ///////////////////////////////////////////
 
 *n_transformed = *n_orig;
 
 //Normalizes the normal
 normalize(n_transformed);
 
 double TinvTrans[4][4] = {{1.0, 0.0, 0.0, 0.0},
                    {0.0, 1.0, 0.0, 0.0},
                    {0.0, 0.0, 1.0, 0.0},
                    {0.0, 0.0, 0.0, 1.0}};
 //Gets the transpose of the inverse matrix
 matTranspose(obj->Tinv, TinvTrans);
 
 //Transforms the normal using the inverse object transformation
 matVecMult(TinvTrans, n_transformed);
 
 //Normalizes the normal after it is transformed
 normalize(n_transformed);

}

/////////////////////////////////////////////
// Object management section
/////////////////////////////////////////////
void insertObject(struct object3D *o, struct object3D **list)
{
 if (o==NULL) return;
 // Inserts an object into the object list.
 if (*(list)==NULL)
 {
  *(list)=o;
  (*(list))->next=NULL;
 }
 else
 {
  o->next=(*(list))->next;
  (*(list))->next=o;
 }
}

struct object3D *newPlane(double ra, double rd, double rs, double rg, double rt, double r, double g, double b, double alpha, double r_index, double shiny)
{
 // Intialize a new plane with the specified parameters:
 // ra, rd, rs, rg - Albedos for the components of the Phong model
 // r, g, b, - Colour for this plane
 // alpha - Transparency, must be set to 1 unless you are doing refraction
 // r_index - Refraction index if you are doing refraction.
 // shiny - Exponent for the specular component of the Phong model
 //
 // The plane is defined by the following vertices (CCW)
 // (1,1,0), (-1,1,0), (-1,-1,0), (1,-1,0)
 // With normal vector (0,0,1) (i.e. parallel to the XY plane)

 struct object3D *plane=(struct object3D *)calloc(1,sizeof(struct object3D));

 if (!plane) fprintf(stderr,"Unable to allocate new plane, out of memory!\n");
 else
 {
  plane->alb.ra=ra;
  plane->alb.rd=rd;
  plane->alb.rs=rs;
  plane->alb.rg=rg;
  plane->alb.rt=rt;
  plane->col.R=r;
  plane->col.G=g;
  plane->col.B=b;
  plane->alpha=alpha;
  plane->r_index=r_index;
  plane->shinyness=shiny;
  plane->intersect=&planeIntersect;
  plane->surfaceCoords=&planeCoordinates;
  plane->randomPoint=&planeSample;
  plane->texImg=NULL;
  plane->photonMap=NULL;
  plane->normalMap=NULL;
  memcpy(&plane->T[0][0],&eye4x4[0][0],16*sizeof(double));
  memcpy(&plane->Tinv[0][0],&eye4x4[0][0],16*sizeof(double));
  plane->textureMap=&texMap;
  plane->frontAndBack=1;
  plane->photonMapped=0;
  plane->normalMapped=0;
  plane->isCSG=0;
  plane->isLightSource=0;
  plane->next=NULL;
}
 return(plane);
}

struct object3D *newSphere(double ra, double rd, double rs, double rg, double rt, double r, double g, double b, double alpha, double r_index, double shiny)
{
 // Intialize a new sphere with the specified parameters:
 // ra, rd, rs, rg - Albedos for the components of the Phong model
 // r, g, b, - Colour for this plane
 // alpha - Transparency, must be set to 1 unless you are doing refraction
 // r_index - Refraction index if you are doing refraction.
 // shiny -Exponent for the specular component of the Phong model
 //
 // This is assumed to represent a unit sphere centered at the origin.
 //

 struct object3D *sphere=(struct object3D *)calloc(1,sizeof(struct object3D));

 if (!sphere) fprintf(stderr,"Unable to allocate new sphere, out of memory!\n");
 else
 {
  sphere->alb.ra=ra;
  sphere->alb.rd=rd;
  sphere->alb.rs=rs;
  sphere->alb.rg=rg;
  sphere->alb.rt=rt;
  sphere->col.R=r;
  sphere->col.G=g;
  sphere->col.B=b;
  sphere->alpha=alpha;
  sphere->r_index=r_index;
  sphere->shinyness=shiny;
  sphere->intersect=&sphereIntersect;
  sphere->surfaceCoords=&sphereCoordinates;
  sphere->randomPoint=&sphereSample;
  sphere->texImg=NULL;
  sphere->photonMap=NULL;
  sphere->normalMap=NULL;
  memcpy(&sphere->T[0][0],&eye4x4[0][0],16*sizeof(double));
  memcpy(&sphere->Tinv[0][0],&eye4x4[0][0],16*sizeof(double));
  sphere->textureMap=&texMap;
  sphere->frontAndBack=0;
  sphere->photonMapped=0;
  sphere->normalMapped=0;
  sphere->isCSG=0;
  sphere->isLightSource=0;
  sphere->next=NULL; }
 return(sphere);
}

struct object3D *newCyl(double ra, double rd, double rs, double rg, double rt, double r, double g, double b, double alpha, double r_index, double shiny)
{
 ///////////////////////////////////////////////////////////////////////////////////////
 // TO DO:
 //	Complete the code to create and initialize a new cylinder object.
 ///////////////////////////////////////////////////////////////////////////////////////
 
 //Assuming the base cylinder will be a right circular cylinder with 
 //p0(a, b) = (cos(a), sin(a), b) for 0 <= a < 2pi, 0 <= b <= 1
 //Cylinder with center of base at origin and going up into z
  struct object3D *cylinder=(struct object3D *)calloc(1,sizeof(struct object3D));
   if (!cylinder) fprintf(stderr,"Unable to allocate new cylinder, out of memory!\n");
 else
 {
  cylinder->alb.ra=ra;
  cylinder->alb.rd=rd;
  cylinder->alb.rs=rs;
  cylinder->alb.rg=rg;
  cylinder->alb.rt=rt;
  cylinder->col.R=r;
  cylinder->col.G=g;
  cylinder->col.B=b;
  cylinder->alpha=alpha;
  cylinder->r_index=r_index;
  cylinder->shinyness=shiny;
  cylinder->intersect=&cylIntersect;
  cylinder->surfaceCoords=&cylCoordinates;
  cylinder->randomPoint=&cylSample;
  cylinder->texImg=NULL;
  cylinder->photonMap=NULL;
  cylinder->normalMap=NULL;
  memcpy(&cylinder->T[0][0],&eye4x4[0][0],16*sizeof(double));
  memcpy(&cylinder->Tinv[0][0],&eye4x4[0][0],16*sizeof(double));
  cylinder->textureMap=&texMap;
  cylinder->frontAndBack=0;
  cylinder->photonMapped=0;
  cylinder->normalMapped=0;
  cylinder->isCSG=0;
  cylinder->isLightSource=0;
  cylinder->next=NULL; }
 return(cylinder);
}

struct object3D *newImplicit(void (*distance)(struct point3D *point, double *value), void (*normal)(struct point3D *p, struct point3D *n), double ra, double rd, double rs, double rg, double rt, double r, double g, double b, double alpha, double R_index, double shiny)
{
	struct object3D *implicit=(struct object3D *)calloc(1,sizeof(struct object3D));
   if (!implicit) fprintf(stderr,"Unable to allocate new cylinder, out of memory!\n");
 else
 {
  implicit->alb.ra=ra;
  implicit->alb.rd=rd;
  implicit->alb.rs=rs;
  implicit->alb.rg=rg;
  implicit->alb.rt=rt;
  implicit->col.R=r;
  implicit->col.G=g;
  implicit->col.B=b;
  implicit->alpha=alpha;
  implicit->r_index=R_index;
  implicit->shinyness=shiny;
  implicit->intersect=&implicitIntersect;
  implicit->surfaceCoords=&implicitCoordinates;
  implicit->implicit=distance;
  implicit->implicitNormal=normal;
  implicit->randomPoint=&implicitSample;
  implicit->texImg=NULL;
  implicit->photonMap=NULL;
  implicit->normalMap=NULL;
  memcpy(&implicit->T[0][0],&eye4x4[0][0],16*sizeof(double));
  memcpy(&implicit->Tinv[0][0],&eye4x4[0][0],16*sizeof(double));
  implicit->textureMap=&texMap;
  implicit->frontAndBack=0;
  implicit->photonMapped=0;
  implicit->normalMapped=0;
  implicit->isCSG=0;
  implicit->isLightSource=0;
  implicit->next=NULL; }
 return(implicit);
}

//Shape describes what shape it uses. 0 for plane, 1 for sphere, 2 for cylinder
struct object3D *newCSG(int shape, int operation, double ra, double rd, double rs, double rg, double rt, double r, double g, double b, double alpha, double r_index, double shiny)
{
	struct object3D *newObj;
	if(shape == 0){
		newObj = newPlane(ra, rd, rs, rg, rt, r, g, b, alpha, r_index, shiny);
	}else if(shape == 1){
		newObj = newPlane(ra, rd, rs, rg, rt, r, g, b, alpha, r_index, shiny);
	}else if(shape == 2){
		newObj = newPlane(ra, rd, rs, rg, rt, r, g, b, alpha, r_index, shiny);
	}else{
		printf("Invalid shape type. 0 = plane, 1 = sphere, 2 = cylinder");
		return NULL;
	}
	newObj->isCSG=1;
	newObj->CSGOperation=operation;
	return(newObj);
	
}
///////////////////////////////////////////////////////////////////////////////////////
// TO DO:
//	Complete the functions that compute intersections for the canonical plane
//      and canonical sphere with a given ray. This is the most fundamental component
//      of the raytracer.
///////////////////////////////////////////////////////////////////////////////////////
void planeIntersect(struct object3D *plane, struct ray3D *ray, double *lambda, struct point3D *p, struct point3D *n, double *a, double *b)
{
	//printf("Object color plane: (%f, %f, %f)\n", plane->col.R, plane->col.G, plane->col.B);
 // Computes and returns the value of 'lambda' at the intersection
 // between the specified ray and the specified canonical plane.
 // The plane is defined by the following vertices (CCW)
 // (1,1,0), (-1,1,0), (-1,-1,0), (1,-1,0)
 // With normal vector (0,0,1) (i.e. parallel to the XY plane)
 /////////////////////////////////
 // TO DO: Complete this function.
 /////////////////////////////////
 
 // Bottom left is now the point on the plane p
 // vector A is the vertical vector for the planar patch of height 2 
 // vector B is the horizontal vector for the planar patch of width 2
 // normal is the normal to a plane on the x-y axis
 struct point3D bottomLeft = { .px = -1, .py = -1, .pz = 0, .pw = 1}; 
 struct point3D vectorA = { .px = 0, .py = 2, .pz = 0, .pw = 1};
 struct point3D vectorB = { .px = 2, .py = 0, .pz = 0, .pw = 1};
 struct point3D planeNormal = { .px = 0, .py = 0, .pz = 1, .pw = 0};

 
 struct ray3D transformedRay;
 rayTransform(ray, &transformedRay , plane);
 
 //If we represent the ray as l(\lambda) = p + (\lambda)d
 //Then the intersect point between the ray and the plane is when (l(\lambda) - p0)n = 0
 //This is equal to ((\lambda)d)*n = (p0 - p)*n
 //=> (\lambda) = ((p0 - p)*n)/(d*n)
 struct point3D pointDifference = bottomLeft;
 //pointdifference = pointdifference - transformedray
 subVectors(&transformedRay.p0, &pointDifference); 
 double lambdaValue = dot(&pointDifference, &planeNormal)/dot(&transformedRay.d, &planeNormal);
 
 
 //Calculation for POI of unit plane
 struct point3D normPOI = {.px = transformedRay.p0.px + transformedRay.d.px*lambdaValue,
						   .py = transformedRay.p0.py + transformedRay.d.py*lambdaValue,
						   .pz = transformedRay.p0.pz + transformedRay.d.pz*lambdaValue,
						   .pw = 1};
 //Calculation for POI of world plane
 struct point3D POI = normPOI;
 matVecMult(plane->T, &POI);
 //For unit plane POI
 //POI.y = 2(\alpha) - 1 => (\alpha) = (POI.y + 1)/2
 //POI.x = 2(\beta) - 1 => (\beta) = (POI.x + 1)/2
 double beta = (normPOI.px + 1)/2;
 double alpha = (normPOI.py + 1)/2;
 *a = 0;
 *b = 0;
 
 //Only assign lambda if value is within planar patch
 if(beta <= 1 && beta >= 0 && alpha <= 1 && alpha >= 0){
	 if(plane->normalMap != NULL){
		normalMap(plane->normalMap, alpha, beta, &planeNormal.px, &planeNormal.py, &planeNormal.pz);
		normalize(&planeNormal);
		//printf("(%f,%f,%f)\n", planeNormal.px, planeNormal.py, planeNormal.pz);
		
	 }
	 normalTransform(&planeNormal, &planeNormal, plane);
	 //Make sure the normal is greater or equals to 90 degrees to the incident ray.
	 if(dot(&ray->d, &planeNormal) > 0){
		//Invert the direction of the normal if it is not
		scalarMultVector(&planeNormal, -1);
	 }
	 
	*n = planeNormal;
	*lambda = lambdaValue;
	*p = POI;
	*a = alpha;
	*b = beta;
 } else {
	*lambda = -1;
 }
 if(*a < -1 || *a > 1 || *b < -1 || *b > 1){
	printf("Plane: %f,%f\n", *a, *b);
 }
}

void sphereIntersect(struct object3D *sphere, struct ray3D *ray, double *lambda, struct point3D *p, struct point3D *n, double *a, double *b)
{
	if(ray->p0.px + 0.193628 <PRECISION && ray->p0.py + 0.005090 < PRECISION && ray->p0.pz < PRECISION && 
	ray->d.px + 0.190095 < PRECISION && ray->d.py + 0.004997 < PRECISION && ray->d.pz - 0.981753 < PRECISION){
		//printf("Good\n");
	}
	//printf("Object color sphere: (%f, %f, %f)\n", sphere->col.R, sphere->col.G, sphere->col.B);
 // Computes and returns the value of 'lambda' at the intersection
 // between the specified ray and the specified canonical sphere.

 /////////////////////////////////
 // TO DO: Complete this function.
 /////////////////////////////////
 //Transform the ray based on the inverse transformation matrix of the sphere to get a ray for a unit sphere
 struct ray3D transformedRay;
 rayTransform(ray, &transformedRay , sphere);
 normalize(&transformedRay.d);

 //Intersection with sphere is ||p - c||^2 - R^2 = 0
 //With a unit sphere at origin is ||p||^2 - 1 = 0
 //Substituting the ray as our point, we have ||p + \lambda d||^2 - 1 = 0
 //Equals to: (px + (\lambda)dx)^2 + (py + (\lambda)dy)^2 + (pz + (\lambda)dz)^2 - 1 = 0
 //Equals to: + (\lambda)^2(dx^2 + dy^2 + dz^2) + 2(\lambda)(pxdx + pydy + pzdz) + (px^2 + py^2 + pz^2 - 1) = 0
 //A = (dx^2 + dy^2 + dz^2), B = 2(pxdx + pydy + pzdz), C = (px^2 + py^2 + pz^2 - 1)
 double A = transformedRay.d.px*transformedRay.d.px + 
			transformedRay.d.py*transformedRay.d.py + 
			transformedRay.d.pz*transformedRay.d.pz;
			
 double B = 2*(transformedRay.p0.px*transformedRay.d.px + 
			transformedRay.p0.py*transformedRay.d.py + 
			transformedRay.p0.pz*transformedRay.d.pz);
			
 double C = transformedRay.p0.px*transformedRay.p0.px + 
			transformedRay.p0.py*transformedRay.p0.py + 
			transformedRay.p0.pz*transformedRay.p0.pz - 1;
//Check how many intersections with b^2 - 4ac
 double trueLambdaValue = -1;

 if(B*B < 4*A*C){//No intersections
	//No need to do anything
 } else if(B*B - 4*A*C  < PRECISION && B*B - 4*A*C > -PRECISION){//1 Intersection
	double lambdaValue = -B/(2*A);
	//Take the intersection if it is positive, else return no intersection
	if(lambdaValue > 0){
		trueLambdaValue = lambdaValue;
	}
 } else {//2 Intersections
	//printf("lambdavalues: %f, %f\n", lambdaValue1, lambdaValue2);
	 double lambdaValue1 = (-B + sqrt(B*B - 4*A*C))/(2*A);
	 double lambdaValue2 = (-B - sqrt(B*B - 4*A*C))/(2*A);
	 //printf("%f, %f\n", lambdaValue1, lambdaValue2);
	//Take the closest positive intersection
	if(lambdaValue1 <= PRECISION && lambdaValue2 <= PRECISION){
		//Take no lambda if they are negative
	}else if(lambdaValue1 <= PRECISION){
		trueLambdaValue = lambdaValue2;
	} else if(lambdaValue2 <= PRECISION){
		trueLambdaValue = lambdaValue1;
	} else {
		trueLambdaValue = (lambdaValue1 < lambdaValue2) ? lambdaValue1 : lambdaValue2;
	}
 }
 *lambda = trueLambdaValue;
 
 //Calculate POI, normal, phi and theta only if a valid lambda exists
 if(*lambda > 0){
	//printf("Sphere Lambda: %f\n", trueLambdaValue);
	//POI is p0 + (\lambda)d;
	
	//a is theta, b is phi
	//Corresponds to (cos(a)sin(b), sin(a)sin(b), cos(b)) for a unit sphere;
	//POI on the unit sphere is
	struct point3D unitPOI = {.px = transformedRay.p0.px + trueLambdaValue*transformedRay.d.px, 
							  .py = transformedRay.p0.py + trueLambdaValue*transformedRay.d.py, 
							  .pz = transformedRay.p0.pz + trueLambdaValue*transformedRay.d.pz,
							  .pw = 1};
	
	//mPOI is POI on original object
	//mPOI gives correct POI for affine transformed sphere
	struct point3D mPOI = unitPOI;
	matVecMult(sphere->T, &mPOI);
	*p=mPOI;

	//printf("UnitPOI:(%f, %f, %f), POI: (%f, %f, %f), matPOI:(%f, %f, %f)\n", unitPOI.px, unitPOI.py, unitPOI.pz, p->px, p->py, p->pz, mPOI.px, mPOI.py, mPOI.pz);
	
	//Implicit form for unit sphere is x^2 + y^2 + z^2 - 1 = 0
	//So, normal n = (2x, 2y, 2z);
	//printmatrix(sphere->T);
	//printmatrix(sphere->Tinv);
	struct point3D unitNormal = {.px = 2*unitPOI.px, 
								 .py = 2*unitPOI.py, 
								 .pz = 2*unitPOI.pz, 
								 .pw = 1};
	//beta = arccos(z)
	//alpha = arcsin(y/sin(beta)) or arccos(x/sin(beta))
	double beta = acos(unitPOI.pz);
	double alpha = 0;
	if(sin(beta) > PRECISION && unitPOI.py/sin(beta) <= 1 && unitPOI.py/sin(beta) >= -1){
		alpha = asin(unitPOI.py/sin(beta));	
	}
	if(isunordered(alpha, beta)){
		printf("For some unknown reason, alpha is not a number!\n");
		printf("Beta0: %f, %f\n", unitPOI.py, beta);
		alpha = 0;
	}
	//Normalize a and b
	//b = [0, pi]
	//b/pi = [0, 1]
	//a = [-pi, pi]
	//(a + pi)/2pi = [0,1]
	*b = beta/M_PI;
	*a = (alpha + M_PI)/(2*M_PI);
	if(sphere->normalMap != NULL){
		normalMap(sphere->normalMap, *a, *b, &unitNormal.px, &unitNormal.py, &unitNormal.pz);
		//Uses spherical coordinates originating at theta = phi = 0, which corresponds to the (0,0,1) vector
		struct object3D *o = newSphere(.2,.95,.75,.75,1,.75,.95,.55,0,1,6);
		RotateZ(o, -beta);
		RotateY(o, -alpha);
		matVecMult(o->T, &unitNormal);
		free(o);
	}
	normalize(&unitNormal);
	//printf("UnitNormal: (%f, %f, %f)\n", unitNormal.px, unitNormal.py, unitNormal.pz);
	normalTransform(&unitNormal, &unitNormal, sphere);
	//printf("UnitNormalFormula: (%f, %f, %f)\n", unitNormal.px, unitNormal.py, unitNormal.pz);
	*n = unitNormal;
 }
	if(*a <  -1 || *a > 1 || *b < -1 || *b > 1){
		printf("Sphere: %f,%f\n", *a, *b);
	}
}

void cylIntersect(struct object3D *cylinder, struct ray3D *ray, double *lambda, struct point3D *p, struct point3D *n, double *a, double *b)
{
 // Computes and returns the value of 'lambda' at the intersection
 // between the specified ray and the specified canonical cylinder.
 
 /////////////////////////////////
 // TO DO: Complete this function.
 /////////////////////////////////
 //Transform the ray so that the ray is working on a unit cylinder
 struct ray3D transformedRay;
 rayTransform(ray, &transformedRay, cylinder);
 normalize(&transformedRay.d);

 //Assume unit cylinder is right circle cylinder with center of the two circles at (0,0,0) and (0,0,1)
 //Computing cylinder intersection with long body and 2 end caps
 //Computing intersection at the end caps, where endcaps are planes normal to the z-axis at z=1 and z=0
 //b is the height [0,1], a is the position on the circle [0, 2pi]
 struct point3D normalz0 = {.px = 0, .py = 0, .pz = -1, .pw = 0};
 struct point3D normalz1 = {.px = 0, .py = 0, .pz = 1, .pw = 0};
 struct point3D z0 = {.px = 0, .py = 0, .pz = 0, .pw = 1};
 struct point3D z1 = {.px = 0, .py = 0, .pz = 1, .pw = 1};
 
 //If we represent the ray as l(\lambda) = p + (\lambda)d
 //Then the intersect point between the ray and the top and bottom face of the cylinder is when (l(\lambda) - p0)n = 0
 //This is equal to ((\lambda)d)*n = (p0 - p)*n
 //=> (\lambda) = ((p0 - p)*n)/(d*n)
 struct point3D pointDifference = z0;
 subVectors(&transformedRay.p0, &pointDifference);
 double lambdaValueZ0 = dot(&pointDifference, &normalz0)/dot(&transformedRay.d, &normalz0);
 
 pointDifference = z1;
 subVectors(&transformedRay.p0, &pointDifference);
 double lambdaValueZ1 = dot(&pointDifference, &normalz1)/dot(&transformedRay.d, &normalz1);
 
 struct point3D UnitPOIz0 = {.px = transformedRay.p0.px + lambdaValueZ0*transformedRay.d.px, 
							 .py = transformedRay.p0.py + lambdaValueZ0*transformedRay.d.py, 
							 .pz = transformedRay.p0.pz + lambdaValueZ0*transformedRay.d.pz,
							 .pw = 1};
 struct point3D UnitPOIz1 = {.px = transformedRay.p0.px + lambdaValueZ1*transformedRay.d.px, 
							 .py = transformedRay.p0.py + lambdaValueZ1*transformedRay.d.py, 
							 .pz = transformedRay.p0.pz + lambdaValueZ1*transformedRay.d.pz,
							 .pw = 1};
 
 //Checks if the ray intersects the endcaps at a valid location
 //They intersect at an invalid location if (x^2 + y^2) > 1
 if((UnitPOIz0.px*UnitPOIz0.px + UnitPOIz0.py*UnitPOIz0.py) > 1){
	lambdaValueZ0 = -1;
 }
 
 if((UnitPOIz1.px*UnitPOIz1.px + UnitPOIz1.py*UnitPOIz1.py) > 1){
	lambdaValueZ1 = -1;
 }
 
 //printmatrix(cylinder->T);
 //printmatrix(cylinder->Tinv);
 //printf("transformedRay:(%f,%f,%f)(%f,%f,%f)\n", transformedRay.p0.px, transformedRay.p0.py, transformedRay.p0.pz, transformedRay.d.px, transformedRay.d.py, transformedRay.d.pz);
 
 //A ray will intersect with the length of cylinder if it intersects with a unit circle on the x-y plane and the z-value is between 0 and 1.
 //With the ray as our point, we have for intersection of a unit circle||p + \lambda d||^2 - 1 = 0
 //Equals to: (px + (\lambda)dx)^2 + (py + (\lambda)dy)^2 - 1 = 0
 //Equals to: + (\lambda)^2(dx^2 + dy^2) + 2(\lambda)(pxdx + pydy) + (px^2 + py^2 - 1) = 0
 //A = (dx^2 + dy^2), B = 2(pxdx + pydy), C = (px^2 + py^2 - 1)
 double A = transformedRay.d.px*transformedRay.d.px + 
			transformedRay.d.py*transformedRay.d.py;
			
 double B = 2*(transformedRay.p0.px*transformedRay.d.px + 
			transformedRay.p0.py*transformedRay.d.py);
			
 double C = transformedRay.p0.px*transformedRay.p0.px + 
			transformedRay.p0.py*transformedRay.p0.py - 1;
 //printmatrix(cylinder->T);
 //printmatrix(cylinder->Tinv);
 //printf("Ray(%f,%f,%f)(%f,%f,%f)\n", transformedRay.p0.px, transformedRay.p0.py, transformedRay.p0.pz, transformedRay.d.px, transformedRay.d.py, transformedRay.d.pz);
 //printf("A,B,C: (%f,%f,%f)\n", A, B, C);
 //Check how many intersections with b^2 - 4ac
 double lambdaValueC1 = -1;
 double lambdaValueC2 = -1;
 if(B*B < 4*A*C){//No intersections
	//No need to do anything
	//printf("0");
 } else if(B*B - 4*A*C  < PRECISION && B*B - 4*A*C > -PRECISION){//1 Intersection
	//Take the intersection if it is positive, else return no intersection
	//printf("1");
	double testLambda = -B/(2*A);
	if(testLambda > 0){
		lambdaValueC1 = testLambda;
	}
 } else {//2 Intersections
	double lambdaValue1 = (-B + sqrt(B*B - 4*A*C))/(2*A);
	double lambdaValue2 = (-B - sqrt(B*B - 4*A*C))/(2*A);
	//printf("2");
	//printf("Lambdas:(%f,%f)\n", lambdaValue1, lambdaValue2);
	if(lambdaValue1 > PRECISION){
		lambdaValueC1 = lambdaValue1;
	}
	
	if(lambdaValue2 > PRECISION){
		lambdaValueC2 = lambdaValue2;
	}
 }
 
  //printf("Lambdas:(%f,%f,%f,%f)\n",lambdaValueC1, lambdaValueC2,lambdaValueZ0,lambdaValueZ1);
 
 //printf("Lambdas: ")
 //Checks to make sure z-value of intersections are between 0 and 1
 if(lambdaValueC1 > 0){
	double zval = transformedRay.p0.pz + lambdaValueC1*transformedRay.d.pz;
	if(zval < 0 || zval > 1){
		lambdaValueC1 = -1;
	}
 }
 
 if(lambdaValueC2 > 0){
	double zval = transformedRay.p0.pz + lambdaValueC2*transformedRay.d.pz;
	if(zval < 0 || zval > 1){
		lambdaValueC2 = -1;
	}
 }
 //printf("Lambdas:(%f,%f,%f,%f)\n",lambdaValueC1, lambdaValueC2,lambdaValueZ0,lambdaValueZ1);
 //Takes the min value between z0, z1, c1, and c2 that is greater than 0.
 if(lambdaValueZ0 <= 0 && lambdaValueZ1 <= 0 && lambdaValueC1 <= 0 && lambdaValueC2 <= 0){
	double negative1 = -1;
	*lambda = negative1;
 } else {
	double lambdaValue = lambdaValueZ0;
	if(lambdaValue <= 0 || ((lambdaValueZ1 < lambdaValue) && lambdaValueZ1 >= 0)){
		lambdaValue = lambdaValueZ1;
	}
	if(lambdaValue <= 0 || ((lambdaValueC1 < lambdaValue) && lambdaValueC1 >= 0)){
		lambdaValue = lambdaValueC1;
	}
	if(lambdaValue <= 0 || ((lambdaValueC2 < lambdaValue) && lambdaValueC2 >= 0)){
		lambdaValue = lambdaValueC2;
	}
	*lambda = lambdaValue;
 }
 //printf("Lambda:%f", *lambda);
 if(*lambda > 0){
	struct point3D unitPOI = {.px = transformedRay.p0.px + transformedRay.d.px* *lambda, 
							  .py = transformedRay.p0.py + transformedRay.d.py* *lambda, 
							  .pz = transformedRay.p0.pz + transformedRay.d.pz* *lambda, 
							  .pw = 1};
	struct point3D POI = unitPOI;
	*p = POI;
	matVecMult(cylinder->T, &POI);
	struct point3D norm;
	if(unitPOI.pz < 1 + PRECISION && unitPOI.pz > 1-PRECISION){
		//Top face
		norm = normalz1;
		if(cylinder->normalMap != NULL){
			normalMap(cylinder->normalMap, (asin(unitPOI.py) + M_PI/2)/M_PI, unitPOI.pz, &norm.px, &norm.py, &norm.pz);
		}
	} else if(unitPOI.pz < PRECISION && unitPOI.pz > -PRECISION){
		//Bottom face
	    norm = normalz0;
		if(cylinder->normalMap != NULL){
			normalMap(cylinder->normalMap, (asin(unitPOI.py) + M_PI/2)/M_PI, unitPOI.pz, &norm.px, &norm.py, &norm.pz);
			norm.pz = -norm.pz;
		}
	} else {
		//Side face
		norm = {.px = unitPOI.px, .py = unitPOI.py, .pz = unitPOI.pz, .pw = 0};
		if(cylinder->normalMap != NULL){
			normalMap(cylinder->normalMap, (asin(unitPOI.py) + M_PI/2)/M_PI, unitPOI.pz, &norm.px, &norm.py, &norm.pz);
			//Uses spherical coordinates originating at theta = phi = 0, which corresponds to the (0,0,1) vector
			struct object3D *o = newSphere(.2,.95,.75,.75,1,.75,.95,.55,0,1,6);
			RotateZ(o, -M_PI/2);
			RotateY(o, -asin(-norm.py*2/M_PI));
			matVecMult(o->T, &norm);
			free(o);
		}
	}
	normalTransform(&norm, n, cylinder);
	*b = unitPOI.pz;
	*a = asin(unitPOI.py);
	//Normalize a and b
	//b = [0, 1], so it is fine
	//a = [-pi/2, pi/2], so add pi/2 and divide by pi
	*a = ((*a) + M_PI/2)/M_PI;
 }
	if(*a+1 < PRECISION || *a -1 > PRECISION || *b+1 < PRECISION || *b-1 > PRECISION){
		printf("Cylinder: %f,%f\n", *a, *b);
	}
}

void implicitIntersect(struct object3D *implicit, struct ray3D *ray, double *lambda, struct point3D *p, struct point3D *n, double *a, double *b)
{
	struct point3D point;
	int binarySearchSteps = 1;
	double value0;
	double value1;
	double valuemid;
	
	//Transform the ray first
	struct ray3D transformedRay;
	rayTransform(ray, &transformedRay, implicit);
	normalize(&transformedRay.d);
	
	//printf("Ray:(%f,%f,%f) +(%f,%f,%f)\n", ray->p0.px, ray->p0.py, ray->p0.pz, ray->d.px, ray->d.py, ray->d.pz);
	//printf("TRay:(%f,%f,%f) +(%f,%f,%f)\n", transformedRay.p0.px, transformedRay.p0.py, transformedRay.p0.pz, transformedRay.d.px, transformedRay.d.py, transformedRay.d.pz);
	*lambda = 0;
	computePoint(&transformedRay, *lambda, &point);
	implicit->implicit(&point, &value0);
	int done = 0;
	while(!done){
		//printf("Value: %f\n", value0);
		if(value0 < PRECISION && value0 > -PRECISION){ // If the original guess is on the wall
			done = 1;
		} else{
			*lambda = *lambda + LAMBDAINCREMENT;
			computePoint(&transformedRay, *lambda, &point);
			implicit->implicit(&point, &value1);
			if(value1 < PRECISION && value1 > -PRECISION){ //If it is exactly on the wall
				done = 1;
			}else if(value0*value1 < 0){//If the signs are different, then the wall is somewhere in between
				*lambda = *lambda - LAMBDAINCREMENT/pow(2, binarySearchSteps);
				computePoint(&transformedRay, *lambda, &point);
				implicit->implicit(&point, &valuemid);
				while(binarySearchSteps < 5){//Only want to run binary search a maximum of 5 times to save computation
					binarySearchSteps++;
					if(valuemid == 0){
						binarySearchSteps = 5;
					}
					else{ 
						if(valuemid*value0 < 0){
							*lambda = *lambda - LAMBDAINCREMENT/pow(2, binarySearchSteps);
						}else{
							*lambda = *lambda + LAMBDAINCREMENT/pow(2, binarySearchSteps);
						}
						computePoint(&transformedRay, *lambda, &point);
						implicit->implicit(&point, &valuemid);
					}
					
				}
				done = 1;
			}
			value0 = value1;
		}
		
		if(*lambda > RAYMARCHMAX){
			done = 1;
			*lambda = -1;
		}
	}
	if(*lambda >= 0){
		computePoint(&transformedRay, *lambda, &point);
		//POI has been calculated
		*p = point;
		//Calculates the normal at the unit POI
		implicit->implicitNormal(p, n);
		//Transforms the unit POI to the actual POI
		matVecMult(implicit->T, p);
		normalize(n);
		normalTransform(n, n, implicit);
		*a = -1;
		*b = -1;
		//printf("%f\n", *lambda);
	}
}


//Takes the start of a CSGObject. Only the starts of CSG Objects are added to the object list
void CSGIntersect(struct object3D *node, struct ray3D *ray, double *lambda, struct point3D *p, struct point3D *n, double *a, double *b){
	*lambda = -1;
	int length = 1;
	int arrLength = 0;
	struct object3D *tempNode = node;
	while(tempNode != NULL){
		length++;
		tempNode = tempNode->CSGNext;
	}
	double arr[3*length];//First value in array is where the first lambda, each value after that is how far away from the first lambda it is
						 //Postive means that it travels within an object, and negative means it travels in air
	while(tempNode != NULL){
		struct ray3D *tempRay = ray;
		double tempLambda1 = -1, tempLambda2 = -1, a, b;
		struct point3D p, n;
		tempNode->intersect(tempNode, tempRay, &tempLambda1, &p, &n, &a, &b);
		if(tempLambda1 > 0){
			tempRay->p0 = p;
			tempNode->intersect(tempNode, tempRay, &tempLambda2, &p, &n, &a, &b);
		}
		
		if(tempNode->CSGOperation == -2){//First element
			if(tempLambda1 > 0){
				arr[0] = tempLambda1;
				arrLength++;
				if(tempLambda2 > 0){
					arr[1] = tempLambda2;
					arrLength++;
				}
			}
		}else if(tempNode->CSGOperation == -1 && tempLambda1 > 0 && tempLambda2 > 0){//Minus
			double acc = 0;
			double temparr[length*3];
			int j = 0;
			for(int i = 0; i < arrLength; i++){
				acc += abs(arr[i]);
				if(tempLambda1 > acc || tempLambda1 + tempLambda2 < acc){
					temparr[j] = arr[i];
					j++;
				}else if(tempLambda1 < acc && tempLambda1 > acc - arr[i]){
					temparr[j] = tempLambda1;
					if(arr[i] < 0){
						temparr[j] = -temparr[j];
					}
					j++;
				}else if(tempLambda2+tempLambda1 < acc && tempLambda2+tempLambda1 > acc - arr[i]){
					temparr[j] = tempLambda2;
					j++;
					temparr[j] = acc - tempLambda1 - tempLambda2;
					if(arr[i] < 0){
						temparr[j] = -temparr[j];
					}
					j++;
				}
			}
			memcpy(arr, temparr, sizeof(int)*length*3);
		}else if(tempNode->CSGOperation == 0){//Intersect
			double acc = 0;
			double temparr[length*3];
			int j = 0;
			for(int i = 0; i < arrLength; i++){
				acc += abs(arr[i]);
				if(tempLambda1 < acc && tempLambda1 > acc - arr[i]){
					if(arr[i] < 0){
						temparr[j] = acc;
					}else{
						temparr[j] = tempLambda1;
					}
					j++;
				}else if(tempLambda2+tempLambda1 < acc && tempLambda2+tempLambda1 > acc - arr[i]){
					if(arr[i] < 0){
						temparr[j] = acc - arr[i];
					}else{
						temparr[j] = tempLambda2;
					}
					j++;
				}else if(tempLambda1 < acc && tempLambda1 + tempLambda2 > acc){
					temparr[j] = arr[i];
					j++;
				}
			}
			memcpy(arr, temparr, sizeof(int)*length*3);
		}else if(tempNode->CSGOperation == 1){//Union
			double acc = 0;
			double temparr[length*3];
			int j = 0;
			for(int i = 0; i < arrLength; i++){
				acc += abs(arr[i]);
				if(tempLambda1 > acc || tempLambda1 + tempLambda2 < acc){
					temparr[j] = arr[i];
					j++;
				}else if(tempLambda1 < acc && tempLambda1 > acc - arr[i]){
					temparr[j] = tempLambda1;
					if(arr[i] > 0){
						temparr[j] = -temparr[j];
					}
					j++;
				}else if(tempLambda2+tempLambda1 < acc && tempLambda2+tempLambda1 > acc - arr[i]){
					temparr[j] = tempLambda2;
					j++;
					temparr[j] = acc - tempLambda1 - tempLambda2;
					if(arr[i] > 0){
						temparr[j] = -temparr[j];
					}
					j++;
				}
			}
			memcpy(arr, temparr, sizeof(int)*length*3);
		}
		tempNode = tempNode->CSGNext;
	}
	
	//Takes the first non minus lambda point of intersection
	int i = 0;
	while(arr[i] < 0){
		i++;
	}
	*lambda = arr[i];
}

/////////////////////////////////////////////////////////////////
// Surface coordinates & random sampling on object surfaces
/////////////////////////////////////////////////////////////////
void planeCoordinates(struct object3D *plane, double a, double b, double *x, double *y, double *z)
{
 /////////////////////////////////
 // TO DO: Complete this function.
 /////////////////////////////////

 // Return in (x,y,z) the coordinates of a point on the plane given by the 2 parameters a,b in [0,1].
 // 'a' controls displacement from the left side of the plane, 'b' controls displacement from the
 // bottom of the plane.
 
 //Takes the bottom left (-1, -1, 0) as starting point
 //Change in vertial position is decided by vector (0, 2, 0)
 //Change in horizontal position is decided by vector (2, 0, 0) 
 struct point3D planePoint = {.px = 2*a - 1, .py = 2*b - 1, .pz = 0,	.pw = 1};
 
 //Transforms the point by the transformation matrix of the plane to get the actual point
 matVecMult(plane->T, &planePoint);
 
 *x=planePoint.px;
 *y=planePoint.py;
 *z=planePoint.pz;
 
}

void sphereCoordinates(struct object3D *sphere, double a, double b, double *x, double *y, double *z)
{
 // Return in (x,y,z) the coordinates of a point on the plane given by the 2 parameters a,b.
 // 'a' in [0, 2*PI] corresponds to the spherical coordinate theta
 // 'b' in [-PI/2, PI/2] corresponds to the spherical coordinate phi

 /////////////////////////////////
 // TO DO: Complete this function.
 /////////////////////////////////

 //Point on unit circle centered around origin given parameters a and b
 struct point3D spherePoint = {.px = cos(a)*sin(b), .py = sin(a)*sin(b) , .pz = cos(b) , .pw = 1}; 
 
 //Transform the point based on the transformation matrix
 matVecMult(sphere->T, &spherePoint);
 
 *x=spherePoint.px;
 *y=spherePoint.py;
 *z=spherePoint.pz;
}

void cylCoordinates(struct object3D *cyl, double a, double b, double *x, double *y, double *z)
{
 /////////////////////////////////
 // TO DO: Complete this function.
 ///////////////////////////////// 
 
 // Return in (x,y,z) the coordinates of a point on the plane given by the 2 parameters a,b.
 // 'a' in [0, 2*PI] corresponds to angle theta around the cylinder
 // 'b' in [0, 1] corresponds to height from the bottom
 
 //Assuming a right circular cylinder with height 1 and radius 1 is the unit cylinder
 struct point3D cylinderPoint = {.px = cos(a), .py = sin(a), .pz = b, .pw = 1};
 
 //Transform the point based on the transformation matrix
 matVecMult(cyl->T, &cylinderPoint);
 
 *x=cylinderPoint.px;
 *y=cylinderPoint.py;
 *z=cylinderPoint.pz;
}

void implicitCoordinates(struct object3D *implicit, double a, double b, double *x, double *y, double *z)
{
	//Never used
	*x = 0; *y = 0; *z = 0;
}

void planeSample(struct object3D *plane, double *x, double *y, double *z)
{
 // Returns the 3D coordinates (x,y,z) of a randomly sampled point on the plane
 // Sapling should be uniform, meaning there should be an equal change of gedtting
 // any spot on the plane

 /////////////////////////////////
 // TO DO: Complete this function.
 /////////////////////////////////
 
 //Generate 2 decimal numbers between 0 and 1
 double a = (double)rand()/(double)RAND_MAX; 
 double b = (double)rand()/(double)RAND_MAX;
 //Gets the coordinates of those two numbers on the plane
 planeCoordinates(plane, a, b, x, y, z);
}

void sphereSample(struct object3D *sphere, double *x, double *y, double *z)
{
 // Returns the 3D coordinates (x,y,z) of a randomly sampled point on the sphere
 // Sampling should be uniform - note that this is tricky for a sphere, do some
 // research and document in your report what method is used to do this, along
 // with a reference to your source.
 
 /////////////////////////////////
 // TO DO: Complete this function.
 /////////////////////////////////
 //Generate 2 decimal numbers between 0 and 1
 double a = (double)rand()/(double)RAND_MAX; 
 double b = (double)rand()/(double)RAND_MAX;
 //Multiply b by PI, so the actual range is [0, PI]
 b = b*M_PI;
 //Multiply a by PI, then subtract by PI/2, so the range is [-PI/2, PI/2]
 a = (a*M_PI) - M_PI/2;
 //Gets the coordinates of those two numbers on the sphere
 //printf("Before:(%f,%f,%f) ", *x, *y, *z);
 sphereCoordinates(sphere, b, a, x, y, z); 
 //printf("After:(%f,%f,%f)\n", *x, *y, *z);
}

void cylSample(struct object3D *cyl, double *x, double *y, double *z)
{
 // Returns the 3D coordinates (x,y,z) of a randomly sampled point on the cylinder
 // Sampling should be uniform over the cylinder.

 /////////////////////////////////
 // TO DO: Complete this function.
 /////////////////////////////////   
 //Generate 2 decimal numbers between 0 and 1
 double a = (double)rand()/(double)RAND_MAX; 
 double b = (double)rand()/(double)RAND_MAX;
 //Multiply a by 2PI, so the actual range is [0, 2PI]
 a = a*M_PI*2;
 //Gets the coordinates of those two numbers on the cylinder
 cylCoordinates(cyl, a, b, x, y, z); 
}

void implicitSample(struct object3D *implicit, double *x, double *y, double *z)
{
	//Not Used
	*x = 0; *y = 0; *z = 0;
}

/////////////////////////////////
// Implicit shape definitions
/////////////////////////////////

//Implicit function for unit sphere
 void implicitSphere(struct point3D *point, double *value){
	*value = pow(point->px, 2)+pow(point->py, 2)+pow(point->pz, 2) - 1;
	//printf("(%f,%f,%f):%f\n", point->px, point->py, point->pz, *value);
 }

 //Normal function for implicit unit sphere
 void implicitSphereNormal(struct point3D *point, struct point3D *normal){
	struct point3D n = { .px = 2*point->px, .py = 2*point->py, .pz = 2*point->pz, .pw = 1};
	normalize(&n);
	*normal = n;
 }
 
 //Implicit function for chubbs shape
 void implicitChubbs(struct point3D *point, double *value){
	*value = pow(point->px, 4) + pow(point->py, 4) + pow(point->pz, 4) - pow(point->px, 2) - pow(point->py, 2) - pow(point->pz, 2);
	//printf("(%f,%f,%f):%f\n", point->px, point->py, point->pz, *value);
 }

 //Normal function for chubbs shape
 void implicitChubbsNormal(struct point3D *point, struct point3D *normal){
	struct point3D n = {.px = 4*pow(point->px, 3) - 2*point->px, .py = 4*pow(point->py, 3) - 2*point->py, .pz = 4*pow(point->pz, 3) - 2*point->pz, .pw = 1};
	normalize(&n);
	*normal = n;
 }

 //Implicit function for tangle cube shape
 void implicitTangleCube(struct point3D *point, double *value){
	*value = pow(point->px, 4) - 5*pow(point->px, 2) + pow(point->py, 4) - 5*pow(point->py, 2) + pow(point->pz, 4) - 5*pow(point->pz, 2)+11.8;
 }

 //Normal function for tangle cube shape
 void implicitTangleCubeNormal(struct point3D *point, struct point3D *normal){
	struct point3D n = {.px = 4*pow(point->px, 3) - 10*point->px, .py = 4*pow(point->py, 3) - 10*point->py, .pz = 4*pow(point->pz, 3) - 10*point->pz, .pw = 1};
	normalize(&n);
	*normal = n;
 }

 //Add more definitions however you want	

/////////////////////////////////
// Texture mapping functions
/////////////////////////////////
void loadTexture(struct object3D *o, const char *filename, int type, struct textureNode **t_list)
{
 // Load a texture or normal map image from file and assign it to the
 // specified object. 
 // type:   1  ->  Texture map  (RGB, .ppm)
 //         2  ->  Normal map   (RGB, .ppm)
 //         3  ->  Alpha map    (grayscale, .pgm)
 // Stores loaded images in a linked list to avoid replication
 struct image *im;
 struct textureNode *p;
 
 if (o!=NULL)
 {
  // Check current linked list
  p=*(t_list);
  while (p!=NULL)
  {
   if (strcmp(&p->name[0],filename)==0)
   {
    // Found image already on the list
	printf("%d\n", type);
    if (type==1) o->texImg=p->im;
    else if (type==2) o->normalMap=p->im;
    else o->alphaMap=p->im;
    return;
   }
   p=p->next;
  }    

  // Load this texture image 
  if (type==1||type==2)
   im=readPPMimage(filename);
  else if (type==3)
   im=readPGMimage(filename);

  // Insert it into the texture list
  if (im!=NULL)
  {
   p=(struct textureNode *)calloc(1,sizeof(struct textureNode));
   strcpy(&p->name[0],filename);
   p->type=type;
   p->im=im;
   p->next=NULL; 
   // Insert into linked list
   if ((*(t_list))==NULL)
    *(t_list)=p;
   else
   {
    p->next=(*(t_list))->next;
    (*(t_list))->next=p;
   }
   // Assign to object
   if (type==1) o->texImg=im;
   else if (type==2) o->normalMap=im;
   else o->alphaMap=im;
  }
 
 }  // end if (o != NULL)
}


void texMap(struct image *img, double a, double b, double *R, double *G, double *B)
{
 /*
  Function to determine the colour of a textured object at
  the normalized texture coordinates (a,b).

  a and b are texture coordinates in [0 1].
  img is a pointer to the image structure holding the texture for
   a given object.

  The colour is returned in R, G, B. Uses bi-linear interpolation
  to determine texture colour.
 */

 //////////////////////////////////////////////////
 // TO DO (Assignment 4 only):
 //
 //  Complete this function to return the colour
 // of the texture image at the specified texture
 // coordinates. Your code should use bi-linear
 // interpolation to obtain the texture colour.
 //////////////////////////////////////////////////
 int x = a*(img->sx - 1);
 int y = b*(img->sy - 1);
 if(a < -1 || a > 1 || b < -1 || b > 1){
	//printf("%f, %f\n", a, b);
 }
 //printf("(%f,%f),(%d,%d)\n", a, b, img->sx-1, img->sy-1);
 //printf("(%d,%d,%f,%f)\n", x, y, a*(img->sx - 1), b*(img->sy - 1));
 double *rgbIm = (double *)img->rgbdata;
 *(R)=(*(rgbIm+((x+(y*img->sx))*3)+0));	// Returns black - delete this and
 *(G)=(*(rgbIm+((x+(y*img->sx))*3)+1));	// replace with your code to compute
 *(B)=(*(rgbIm+((x+(y*img->sx))*3)+2));	// texture colour at (a,b)
 //if(a < 0.5 && b < 0.5){
//	printf("(%f,%f,%f)\n", *R,*G,*B);
 //}
 return;
}

void normalMap(struct image *img, double a, double b, double *x, double *y, double *z)
{
 /*
  Function to determine the colour of a textured object at
  the normalized texture coordinates (a,b).

  a and b are texture coordinates in [0 1].
  img is a pointer to the image structure holding the texture for
   a given object.

  The colour is returned in R, G, B. Uses bi-linear interpolation
  to determine texture colour.
 */

 //////////////////////////////////////////////////
 // TO DO (Assignment 4 only):
 //
 //  Complete this function to return the colour
 // of the texture image at the specified texture
 // coordinates. Your code should use bi-linear
 // interpolation to obtain the texture colour.
 //////////////////////////////////////////////////
 int i = a*(img->sx - 1);
 int j = b*(img->sy - 1);
 //printf("%f->%d, %f->%d\n", a, i, b, j);
 //printf("%f->%d, %f->%d\n", a, i, b, j);
 if((i+j*img->sx)*3 < 0){
	printf("Less:%d, %d, %f, %f\n", img->sx, img->sy, a, b);
 }else if((i+j*img->sx)*3 > img->sx*img->sy*3 - 6){
	printf("More?:%d, %d, %f, %f\n", img->sx, img->sy, a, b);
 }
 double *rgbIm = (double *)img->rgbdata;
 *(x)=(*(rgbIm+((i+(j*img->sx))*3)+0))*2 - 1;
 *(y)=(*(rgbIm+((i+(j*img->sx))*3)+1))*2 - 1;
 *(z)=(*(rgbIm+((i+(j*img->sx))*3)+2))*2 - 1;
 return;
}


void alphaMap(struct image *img, double a, double b, double *alpha)
{
 // Just like texture map but returns the alpha value at a,b,
 // notice that alpha maps are single layer grayscale images, hence
 // the separate function.

 //////////////////////////////////////////////////
 // TO DO (Assignment 4 only):
 //
 //  Complete this function to return the alpha
 // value from the image at the specified texture
 // coordinates. Your code should use bi-linear
 // interpolation to obtain the texture colour.
 //////////////////////////////////////////////////
 
 *(alpha)=1;	// Returns 1 which means fully opaque. Replace
 return;	// with your code if implementing alpha maps.
}


/////////////////////////////
// Refractive index stack functions
/////////////////////////////

void push(struct objStack *objstack, double index){
	struct objStack *stack=(struct objStack *)calloc(1, sizeof(struct objStack));
	stack->index = index;
	if(objstack != NULL){
		stack->next = objstack;
	}
	objstack = stack;
};

double pop(struct objStack *objstack){
	if(objstack==NULL){
		return -1;
	}
	double index = objstack->index;
	struct objStack *temp = objstack;
	objstack = objstack->next;
	free(temp);
	return index;
};

double peek(struct objStack *objstack){
	if(objstack==NULL){ //Object is in air
		return 1;
	}
	return objstack->index;
}

/////////////////////////////
// Light sources
/////////////////////////////
void insertPLS(struct pointLS *l, struct pointLS **list)
{
 if (l==NULL) return;
 // Inserts a light source into the list of light sources
 if (*(list)==NULL)
 {
  *(list)=l;
  (*(list))->next=NULL;
 }
 else
 {
  l->next=(*(list))->next;
  (*(list))->next=l;
 }

}

void insertALS(struct areaLS *a, struct areaLS **list){
	
	if(a==NULL) return;
	
	if(*(list)==NULL){
		*(list)=a;
		(*(list))->next=NULL;
	} else{
		a->next=(*(list))->next;
		(*(list))->next=a;
	}
}

void addAreaLight(double sx, double sy, double nx, double ny, double nz,\
                  double tx, double ty, double tz, int N,\
                  double r, double g, double b, struct object3D **o_list, struct pointLS **l_list)
{
 /*
   This function sets up and inserts a rectangular area light source
   with size (sx, sy)
   orientation given by the normal vector (nx, ny, nz)
   centered at (tx, ty, tz)
   consisting of (N) point light sources (uniformly sampled)
   and with colour/intensity (r,g,b) 
   
   Note that the light source must be visible as a uniformly colored rectangle which
   casts no shadows. If you require a lightsource to shade another, you must
   make it into a proper solid box with a back and sides of non-light-emitting
   material
 */

  /////////////////////////////////////////////////////
  // TO DO: (Assignment 4!)
  // Implement this function to enable area light sources
  /////////////////////////////////////////////////////

  // NOTE: The best way to implement area light sources is to random sample from the
  //       light source's object surface within rtShade(). This is a bit more tricky
  //       but reduces artifacts significantly. If you do that, then there is no need
  //       to insert a series of point lightsources in this function.
}

///////////////////////////////////
// Geometric transformation section
///////////////////////////////////

void invert(double *T, double *Tinv)
{
 // Computes the inverse of transformation matrix T.
 // the result is returned in Tinv.

 double *U, *s, *V, *rv1;
 int singFlag, i;

 // Invert the affine transform
 U=NULL;
 s=NULL;
 V=NULL;
 rv1=NULL;
 singFlag=0;

 SVD(T,4,4,&U,&s,&V,&rv1);
 if (U==NULL||s==NULL||V==NULL)
 {
  fprintf(stderr,"Error: Matrix not invertible for this object, returning identity\n");
  memcpy(Tinv,eye4x4,16*sizeof(double));
  return;
 }

 // Check for singular matrices...
 for (i=0;i<4;i++) if (*(s+i)<1e-9) singFlag=1;
 if (singFlag)
 {
  fprintf(stderr,"Error: Transformation matrix is singular, returning identity\n");
  memcpy(Tinv,eye4x4,16*sizeof(double));
  return;
 }

 // Compute and store inverse matrix
 InvertMatrix(U,s,V,4,Tinv);

 free(U);
 free(s);
 free(V);
}

void RotateXMat(double T[4][4], double theta)
{
 // Multiply the current object transformation matrix T in object o
 // by a matrix that rotates the object theta *RADIANS* around the
 // X axis.

 double R[4][4];
 memset(&R[0][0],0,16*sizeof(double));

 R[0][0]=1.0;
 R[1][1]=cos(theta);
 R[1][2]=-sin(theta);
 R[2][1]=sin(theta);
 R[2][2]=cos(theta);
 R[3][3]=1.0;

 matMult(R,T);
}

void RotateX(struct object3D *o, double theta)
{
 // Multiply the current object transformation matrix T in object o
 // by a matrix that rotates the object theta *RADIANS* around the
 // X axis.

 double R[4][4];
 memset(&R[0][0],0,16*sizeof(double));

 R[0][0]=1.0;
 R[1][1]=cos(theta);
 R[1][2]=-sin(theta);
 R[2][1]=sin(theta);
 R[2][2]=cos(theta);
 R[3][3]=1.0;

 matMult(R,o->T);
}

void RotateYMat(double T[4][4], double theta)
{
 // Multiply the current object transformation matrix T in object o
 // by a matrix that rotates the object theta *RADIANS* around the
 // Y axis.

 double R[4][4];
 memset(&R[0][0],0,16*sizeof(double));

 R[0][0]=cos(theta);
 R[0][2]=sin(theta);
 R[1][1]=1.0;
 R[2][0]=-sin(theta);
 R[2][2]=cos(theta);
 R[3][3]=1.0;

 matMult(R,T);
}

void RotateY(struct object3D *o, double theta)
{
 // Multiply the current object transformation matrix T in object o
 // by a matrix that rotates the object theta *RADIANS* around the
 // Y axis.

 double R[4][4];
 memset(&R[0][0],0,16*sizeof(double));

 R[0][0]=cos(theta);
 R[0][2]=sin(theta);
 R[1][1]=1.0;
 R[2][0]=-sin(theta);
 R[2][2]=cos(theta);
 R[3][3]=1.0;

 matMult(R,o->T);
}

void RotateZMat(double T[4][4], double theta)
{
 // Multiply the current object transformation matrix T in object o
 // by a matrix that rotates the object theta *RADIANS* around the
 // Z axis.

 double R[4][4];
 memset(&R[0][0],0,16*sizeof(double));

 R[0][0]=cos(theta);
 R[0][1]=-sin(theta);
 R[1][0]=sin(theta);
 R[1][1]=cos(theta);
 R[2][2]=1.0;
 R[3][3]=1.0;

 matMult(R,T);
}

void RotateZ(struct object3D *o, double theta)
{
 // Multiply the current object transformation matrix T in object o
 // by a matrix that rotates the object theta *RADIANS* around the
 // Z axis.

 double R[4][4];
 memset(&R[0][0],0,16*sizeof(double));

 R[0][0]=cos(theta);
 R[0][1]=-sin(theta);
 R[1][0]=sin(theta);
 R[1][1]=cos(theta);
 R[2][2]=1.0;
 R[3][3]=1.0;

 matMult(R,o->T);
}

void TranslateMat(double T[4][4], double tx, double ty, double tz)
{
 // Multiply the current object transformation matrix T in object o
 // by a matrix that translates the object by the specified amounts.

 double tr[4][4];
 memset(&tr[0][0],0,16*sizeof(double));

 tr[0][0]=1.0;
 tr[1][1]=1.0;
 tr[2][2]=1.0;
 tr[0][3]=tx;
 tr[1][3]=ty;
 tr[2][3]=tz;
 tr[3][3]=1.0;

 matMult(tr,T);
}

void Translate(struct object3D *o, double tx, double ty, double tz)
{
 // Multiply the current object transformation matrix T in object o
 // by a matrix that translates the object by the specified amounts.

 double tr[4][4];
 memset(&tr[0][0],0,16*sizeof(double));

 tr[0][0]=1.0;
 tr[1][1]=1.0;
 tr[2][2]=1.0;
 tr[0][3]=tx;
 tr[1][3]=ty;
 tr[2][3]=tz;
 tr[3][3]=1.0;

 matMult(tr,o->T);
}

void ScaleMat(double T[4][4], double sx, double sy, double sz)
{
 // Multiply the current object transformation matrix T in object o
 // by a matrix that scales the object as indicated.

 double S[4][4];
 memset(&S[0][0],0,16*sizeof(double));

 S[0][0]=sx;
 S[1][1]=sy;
 S[2][2]=sz;
 S[3][3]=1.0;

 matMult(S,T);
}

void Scale(struct object3D *o, double sx, double sy, double sz)
{
 // Multiply the current object transformation matrix T in object o
 // by a matrix that scales the object as indicated.

 double S[4][4];
 memset(&S[0][0],0,16*sizeof(double));

 S[0][0]=sx;
 S[1][1]=sy;
 S[2][2]=sz;
 S[3][3]=1.0;

 matMult(S,o->T);
}

void printmatrix(double mat[4][4])
{
 fprintf(stderr,"Matrix contains:\n");
 fprintf(stderr,"%f %f %f %f\n",mat[0][0],mat[0][1],mat[0][2],mat[0][3]);
 fprintf(stderr,"%f %f %f %f\n",mat[1][0],mat[1][1],mat[1][2],mat[1][3]);
 fprintf(stderr,"%f %f %f %f\n",mat[2][0],mat[2][1],mat[2][2],mat[2][3]);
 fprintf(stderr,"%f %f %f %f\n",mat[3][0],mat[3][1],mat[3][2],mat[3][3]);
}

/////////////////////////////////////////
// Camera and view setup
/////////////////////////////////////////
struct view *setupView(struct point3D *e, struct point3D *g, struct point3D *up, double f, double wl, double wt, double wsize)
{
 /*
   This function sets up the camera axes and viewing direction as discussed in the
   lecture notes.
   e - Camera center
   g - Gaze direction
   up - Up vector
   fov - Fild of view in degrees
   f - focal length
 */
 struct view *c;
 struct point3D *u, *v;

 u=v=NULL;

 // Allocate space for the camera structure
 c=(struct view *)calloc(1,sizeof(struct view));
 if (c==NULL)
 {
  fprintf(stderr,"Out of memory setting up camera model!\n");
  return(NULL);
 }

 // Set up camera center and axes
 c->e.px=e->px;		// Copy camera center location, note we must make sure
 c->e.py=e->py;		// the camera center provided to this function has pw=1
 c->e.pz=e->pz;
 c->e.pw=1;

 // Set up w vector (camera's Z axis). w=-g/||g||
 c->w.px=-g->px;
 c->w.py=-g->py;
 c->w.pz=-g->pz;
 c->w.pw=1;
 normalize(&c->w);

 // Set up the horizontal direction, which must be perpenticular to w and up
 u=cross(&c->w, up);
 normalize(u);
 c->u.px=u->px;
 c->u.py=u->py;
 c->u.pz=u->pz;
 c->u.pw=1;

 // Set up the remaining direction, v=(u x w)  - Mind the signs
 v=cross(&c->u, &c->w);
 normalize(v);
 c->v.px=v->px;
 c->v.py=v->py;
 c->v.pz=v->pz;
 c->v.pw=1;

 // Copy focal length and window size parameters
 c->f=f;
 c->wl=wl;
 c->wt=wt;
 c->wsize=wsize;

 // Set up coordinate conversion matrices
 // Camera2World matrix (M_cw in the notes)
 // Mind the indexing convention [row][col]
 c->C2W[0][0]=c->u.px;
 c->C2W[1][0]=c->u.py;
 c->C2W[2][0]=c->u.pz;
 c->C2W[3][0]=0;

 c->C2W[0][1]=c->v.px;
 c->C2W[1][1]=c->v.py;
 c->C2W[2][1]=c->v.pz;
 c->C2W[3][1]=0;

 c->C2W[0][2]=c->w.px;
 c->C2W[1][2]=c->w.py;
 c->C2W[2][2]=c->w.pz;
 c->C2W[3][2]=0;

 c->C2W[0][3]=c->e.px;
 c->C2W[1][3]=c->e.py;
 c->C2W[2][3]=c->e.pz;
 c->C2W[3][3]=1;

 // World2Camera matrix (M_wc in the notes)
 // Mind the indexing convention [row][col]
 c->W2C[0][0]=c->u.px;
 c->W2C[1][0]=c->v.px;
 c->W2C[2][0]=c->w.px;
 c->W2C[3][0]=0;

 c->W2C[0][1]=c->u.py;
 c->W2C[1][1]=c->v.py;
 c->W2C[2][1]=c->w.py;
 c->W2C[3][1]=0;

 c->W2C[0][2]=c->u.pz;
 c->W2C[1][2]=c->v.pz;
 c->W2C[2][2]=c->w.pz;
 c->W2C[3][2]=0;

 c->W2C[0][3]=-dot(&c->u,&c->e);
 c->W2C[1][3]=-dot(&c->v,&c->e);
 c->W2C[2][3]=-dot(&c->w,&c->e);
 c->W2C[3][3]=1;

 free(u);
 free(v);
 return(c);
}

/////////////////////////////////////////
// Image I/O section
/////////////////////////////////////////
struct image *readPPMimage(const char *filename)
{
 // Reads an image from a .ppm file. A .ppm file is a very simple image representation
 // format with a text header followed by the binary RGB data at 24bits per pixel.
 // The header has the following form:
 //
 // P6
 // # One or more comment lines preceded by '#'
 // 340 200
 // 255
 //
 // The first line 'P6' is the .ppm format identifier, this is followed by one or more
 // lines with comments, typically used to inidicate which program generated the
 // .ppm file.
 // After the comments, a line with two integer values specifies the image resolution
 // as number of pixels in x and number of pixels in y.
 // The final line of the header stores the maximum value for pixels in the image,
 // usually 255.
 // After this last header line, binary data stores the RGB values for each pixel
 // in row-major order. Each pixel requires 3 bytes ordered R, G, and B.
 //
 // NOTE: Windows file handling is rather crotchetty. You may have to change the
 //       way this file is accessed if the images are being corrupted on read
 //       on Windows.
 //
 // readPPMdata converts the image colour information to floating point. This is so that
 // the texture mapping function doesn't have to do the conversion every time
 // it is asked to return the colour at a specific location.
 //

 FILE *f;
 struct image *im;
 char line[1024];
 int sizx,sizy;
 int i;
 unsigned char *tmp;
 double *fRGB;
 int tmpi;
 char *tmpc;

 im=(struct image *)calloc(1,sizeof(struct image));
 if (im!=NULL)
 {
  im->rgbdata=NULL;
  f=fopen(filename,"rb+");
  if (f==NULL)
  {
   fprintf(stderr,"Unable to open file %s for reading, please check name and path\n",filename);
   free(im);
   return(NULL);
  }
  tmpc=fgets(&line[0],1000,f);
  if (strcmp(&line[0],"P6\n")!=0)
  {
   fprintf(stderr,"Wrong file format, not a .ppm file or header end-of-line characters missing\n");
   free(im);
   fclose(f);
   return(NULL);
  }
  fprintf(stderr,"%s\n",line);
  // Skip over comments
  tmpc=fgets(&line[0],511,f);
  while (line[0]=='#')
  {
   fprintf(stderr,"%s",line);
   tmpc=fgets(&line[0],511,f);
  }
  sscanf(&line[0],"%d %d\n",&sizx,&sizy);           // Read file size
  fprintf(stderr,"nx=%d, ny=%d\n\n",sizx,sizy);
  im->sx=sizx;
  im->sy=sizy;

  tmpc=fgets(&line[0],9,f);  	                // Read the remaining header line
  fprintf(stderr,"%s\n",line);
  tmp=(unsigned char *)calloc(sizx*sizy*3,sizeof(unsigned char));
  fRGB=(double *)calloc(sizx*sizy*3,sizeof(double));
  if (tmp==NULL||fRGB==NULL)
  {
   fprintf(stderr,"Out of memory allocating space for image\n");
   free(im);
   fclose(f);
   return(NULL);
  }

  tmpi=fread(tmp,sizx*sizy*3*sizeof(unsigned char),1,f);
  fclose(f);
  // Conversion to floating point
  for (i=0; i<sizx*sizy*3; i++) *(fRGB+i)=((double)*(tmp+i))/255.0;
  free(tmp);
  im->rgbdata=(void *)fRGB;

  return(im);
 }

 fprintf(stderr,"Unable to allocate memory for image structure\n");
 return(NULL);
}

struct image *readPGMimage(const char *filename)
{
 // Just like readPPMimage() except it is used to load grayscale alpha maps. In
 // alpha maps, a value of 255 corresponds to alpha=1 (fully opaque) and 0 
 // correspondst to alpha=0 (fully transparent).
 // A .pgm header of the following form is expected:
 //
 // P5
 // # One or more comment lines preceded by '#'
 // 340 200
 // 255
 //
 // readPGMdata converts the image grayscale data to double floating point in [0,1]. 

 FILE *f;
 struct image *im;
 char line[1024];
 int sizx,sizy;
 int i;
 unsigned char *tmp;
 double *fRGB;
 int tmpi;
 char *tmpc;

 im=(struct image *)calloc(1,sizeof(struct image));
 if (im!=NULL)
 {
  im->rgbdata=NULL;
  f=fopen(filename,"rb+");
  if (f==NULL)
  {
   fprintf(stderr,"Unable to open file %s for reading, please check name and path\n",filename);
   free(im);
   return(NULL);
  }
  tmpc=fgets(&line[0],1000,f);
  if (strcmp(&line[0],"P5\n")!=0)
  {
   fprintf(stderr,"Wrong file format, not a .pgm file or header end-of-line characters missing\n");
   free(im);
   fclose(f);
   return(NULL);
  }
  // Skip over comments
  tmpc=fgets(&line[0],511,f);
  while (line[0]=='#')
   tmpc=fgets(&line[0],511,f);
  sscanf(&line[0],"%d %d\n",&sizx,&sizy);           // Read file size
  im->sx=sizx;
  im->sy=sizy;

  tmpc=fgets(&line[0],9,f);  	                // Read the remaining header line
  tmp=(unsigned char *)calloc(sizx*sizy,sizeof(unsigned char));
  fRGB=(double *)calloc(sizx*sizy,sizeof(double));
  if (tmp==NULL||fRGB==NULL)
  {
   fprintf(stderr,"Out of memory allocating space for image\n");
   free(im);
   fclose(f);
   return(NULL);
  }

  tmpi=fread(tmp,sizx*sizy*sizeof(unsigned char),1,f);
  fclose(f);

  // Conversion to double floating point
  for (i=0; i<sizx*sizy; i++) *(fRGB+i)=((double)*(tmp+i))/255.0;
  free(tmp);
  im->rgbdata=(void *)fRGB;

  return(im);
 }

 fprintf(stderr,"Unable to allocate memory for image structure\n");
 return(NULL);
}

struct image *newImage(int size_x, int size_y)
{
 // Allocates and returns a new image with all zeros. Assumes 24 bit per pixel,
 // unsigned char array.
 struct image *im;

 im=(struct image *)calloc(1,sizeof(struct image));
 if (im!=NULL)
 {
  im->rgbdata=NULL;
  im->sx=size_x;
  im->sy=size_y;
  im->rgbdata=(void *)calloc(size_x*size_y*3,sizeof(unsigned char));
  if (im->rgbdata!=NULL) return(im);
 }
 fprintf(stderr,"Unable to allocate memory for new image\n");
 return(NULL);
}

void imageOutput(struct image *im, const char *filename)
{
 // Writes out a .ppm file from the image data contained in 'im'.
 // Note that Windows typically doesn't know how to open .ppm
 // images. Use Gimp or any other seious image processing
 // software to display .ppm images.
 // Also, note that because of Windows file format management,
 // you may have to modify this file to get image output on
 // Windows machines to work properly.
 //
 // Assumes a 24 bit per pixel image stored as unsigned chars
 //

 FILE *f;

 if (im!=NULL)
  if (im->rgbdata!=NULL)
  {
   f=fopen(filename,"wb+");
   if (f==NULL)
   {
    fprintf(stderr,"Unable to open file %s for output! No image written\n",filename);
    return;
   }
   fprintf(f,"P6\n");
   fprintf(f,"# Output from RayTracer.c\n");
   fprintf(f,"%d %d\n",im->sx,im->sy);
   fprintf(f,"255\n");
   fwrite((unsigned char *)im->rgbdata,im->sx*im->sy*3*sizeof(unsigned char),1,f);
   fclose(f);
   return;
  }
 fprintf(stderr,"imageOutput(): Specified image is empty. Nothing output\n");
}

void deleteImage(struct image *im)
{
 // De-allocates memory reserved for the image stored in 'im'
 if (im!=NULL)
 {
  if (im->rgbdata!=NULL) free(im->rgbdata);
  free(im);
 }
}

void cleanup(struct object3D *o_list, struct pointLS *l_list, struct textureNode *t_list)
{
 // De-allocates memory reserved for the object list and the point light source
 // list. Note that *YOU* must de-allocate any memory reserved for images
 // rendered by the raytracer.
 struct object3D *p, *q;
 struct pointLS *r, *s;
 struct textureNode *t, *u;

 p=o_list;		// De-allocate all memory from objects in the list
 while(p!=NULL)
 {
  q=p->next;
  if (p->photonMap!=NULL)	// If object is photon mapped, free photon map memory
  {
   if (p->photonMap->rgbdata!=NULL) free(p->photonMap->rgbdata);
   free(p->photonMap);
  }
  free(p);
  p=q;
 }

 r=l_list;		// Delete light source list
 while(r!=NULL)
 {
  s=r->next;
  free(r);
  r=s;
 }

 t=t_list;		// Delete texture Images
 while(t!=NULL)
 {
  u=t->next;
  if (t->im->rgbdata!=NULL) free(t->im->rgbdata);
  free(t->im);
  free(t);
  t=u;
 }
}

