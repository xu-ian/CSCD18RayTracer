/*
  CSC D18 - RayTracer code.

  Written Dec. 9 2010 - Jan 20, 2011 by F. J. Estrada
  Freely distributable for adacemic purposes only.

  Uses Tom F. El-Maraghi's code for computing inverse
  matrices. You will need to compile together with
  svdDynamic.c

  You need to understand the code provided in
  this file, the corresponding header file, and the
  utils.c and utils.h files. Do not worry about
  svdDynamic.c, we need it only to compute
  inverse matrices.

  You only need to modify or add code in sections
  clearly marked "TO DO" - remember to check what
  functionality is actually needed for the corresponding
  assignment!

  Last updated: Aug. 2017   - F.J.E.
*/

/*****************************************************************************
* COMPLETE THIS TEXT BOX:
*
* 1) Student Name: Ian Xu		
* 2) Student Name:		
*
* 1) Student number: 1006319208
* 2) Student number:
* 
* 1) UtorID: xuian
* 2) UtorID
* 
* We hereby certify that the work contained here is our own
*
* _______Ian_Xu_______             _____________________
* (sign with your name)            (sign with your name)
********************************************************************************/

#include "utils.h"	// <-- This includes RayTracer.h

#define AANumber 5
#define AREALIGHTSAMPLES 10
#define NUM_THREADS 4

// A couple of global structures and data: An object list, a light list, and the
// maximum recursion depth
struct object3D *object_list;
struct pointLS *light_list;
struct textureNode *texture_list;
int MAX_DEPTH;
struct timespec start, end;
double elapsed;

void buildScene(void)
{//Comment and uncomment lines to render different scenes
#include "buildscene.c"			// Scene definition for MyRender.ppm(Test Scene) 
//#include "buildsceneImplicit.c"	// Scene definition for implicit.ppm
//#include "buildsceneNormal.c"		// Scene definition for NormalMap.ppm
}


//Recursive helper function to determine whether there are shadows on that point by iterating through the light list
//Returns 0 or 1 if it is a point and a value from 0 to 1 if it is an area light source
//Numbers are used to multiply the final col value.
double shadowChecker(struct object3D * colObj, struct point3D *POI, struct point3D *normal, struct pointLS *lightSource){

	//If there are no light sources return false
	if(lightSource == NULL){
		return 0;
	}
	
	//Initializes all the other variables required to use findFirstHit, all except lambda are never used
	double a, b, lambda = -1;
	struct point3D p, p2, n;
	//Point Light Source Calculations
	if(!lightSource->isObj){
		//printf("Point Light Source\n");
		struct point3D d = lightSource->p0;
		subVectors(POI, &d);
		//Initializes the ray from the object to the lightsource starting at the POI
		struct ray3D ray;
		initRay(&ray, POI, &d);

		//Initializes the ray from the lightsource to the POI
		struct object3D *obj = object_list;
		findFirstHit(&ray, &lambda, colObj, &obj, &p, &n, &a, &b, 1);
		//If it hits another object, then a shadow can exist
		if(lambda > 0){
			return shadowChecker(colObj, POI, normal, lightSource->next);
		}
		if(dot(normal, &d) < 0){
			return shadowChecker(colObj, POI, normal, lightSource->next);
		}
		//Otherwise a shadow cannot exist
		return 1;
	}
	//Area Light Source Calculations
	double x, y, z;
	int acc = 0;
	for(int i = 0; i < AREALIGHTSAMPLES; i++){
		//printf("Area light source\n");
		lightSource->obj.randomPoint(&lightSource->obj, &x, &y, &z);
		struct point3D d = {.px = x, .py = y, .pz = z, .pw= 1};
		subVectors(POI, &d);
		struct ray3D ray;
		initRay(&ray, POI, &d);
		struct object3D *obj = object_list;
		findFirstHit(&ray, &lambda, colObj, &obj, &p, &n, &a, &b, 1);
		if(!(lambda > 0 || dot(normal, &d) < 0)){
			acc++;
		}
	}
	if(acc == 0){
		return shadowChecker(colObj, POI, normal, lightSource->next);
	}
	if(acc < AREALIGHTSAMPLES){
		//printf("%d\n", acc);
	}
	return max((double)acc/(double)AREALIGHTSAMPLES, shadowChecker(colObj, POI, normal, lightSource->next));
}

//Calculation for different images given the different components
//Types: 1 = normals, 2 = ambient, 3 = diffuse + ambient + shadows, 4 = specular + shadows, 5 = ambient + diffuse + specular + shadows, 6 = global reflection only, 7 = full without refraction, 
//8 = full with refraction
void colorCalc(struct colourRGB *Itotal, struct colourRGB *ambient, struct colourRGB *diffuse, struct colourRGB *specular, 
			   double shadowConstant, struct colourRGB *Ispec, struct colourRGB *Irefr, struct point3D *n, struct object3D *obj, int type){
	if(type == 1){
		Itotal->R = (n->px + 1)/2; 
		Itotal->G = (n->py + 1)/2; 
		Itotal->B = (n->pz + 1)/2;
	}else if(type == 2){
		Itotal->R = ambient->R;
		Itotal->G = ambient->G;
		Itotal->B = ambient->B;
	}else if(type == 3){
		Itotal->R = ambient->R*0.1 + (diffuse->R*obj->alb.rd)*shadowConstant;
		Itotal->G = ambient->G*0.1 + (diffuse->G*obj->alb.rd)*shadowConstant;
		Itotal->B = ambient->B*0.1 + (diffuse->B*obj->alb.rd)*shadowConstant;
	}if(type == 4){
		Itotal->R = (specular->R*obj->alb.rs)*shadowConstant;
		Itotal->G = (specular->G*obj->alb.rs)*shadowConstant;
		Itotal->B = (specular->B*obj->alb.rs)*shadowConstant;
	}if(type == 5){
		Itotal->R = ambient->R*obj->alb.ra + (diffuse->R*obj->alb.rd + specular->R*obj->alb.rs)*shadowConstant;
		Itotal->G = ambient->G*obj->alb.ra + (diffuse->G*obj->alb.rd + specular->G*obj->alb.rs)*shadowConstant;
		Itotal->B = ambient->B*obj->alb.ra + (diffuse->B*obj->alb.rd + specular->B*obj->alb.rs)*shadowConstant;
	}if(type == 6){
		Itotal->R = Ispec->R;
		Itotal->G = Ispec->G;
		Itotal->B = Ispec->B;
	}else if(type == 7){
		Itotal->R = ambient->R*obj->alb.ra + (diffuse->R*obj->alb.rd + specular->R*obj->alb.rs)*shadowConstant + Ispec->R*obj->alb.rg;
		Itotal->G = ambient->G*obj->alb.ra + (diffuse->G*obj->alb.rd + specular->G*obj->alb.rs)*shadowConstant + Ispec->G*obj->alb.rg;
		Itotal->B = ambient->B*obj->alb.ra + (diffuse->B*obj->alb.rd + specular->B*obj->alb.rs)*shadowConstant + Ispec->B*obj->alb.rg;
	}else if(type == 8){
		//printf("%f, %f, %f\n", Irefr->R, Irefr->G, Irefr->B);
		Itotal->R = obj->alpha*(ambient->R*obj->alb.ra + (diffuse->R*obj->alb.rd + specular->R*obj->alb.rs)*shadowConstant + Ispec->R*obj->alb.rg) + (1.0 - obj->alpha)*Irefr->R*obj->alb.rt;
		Itotal->G = obj->alpha*(ambient->G*obj->alb.ra + (diffuse->G*obj->alb.rd + specular->G*obj->alb.rs)*shadowConstant + Ispec->G*obj->alb.rg) + (1.0 - obj->alpha)*Irefr->G*obj->alb.rt;
		Itotal->B = obj->alpha*(ambient->B*obj->alb.ra + (diffuse->B*obj->alb.rd + specular->B*obj->alb.rs)*shadowConstant + Ispec->B*obj->alb.rg) + (1.0 - obj->alpha)*Irefr->B*obj->alb.rt;
	}
}

//Iteratively goes through each light source and calculates the diffuse value for each point
//For col, 1 = Red, 2 = Green, 3 = Blue
double lightDots(struct pointLS *lightSource, struct object3D *collisionObject, struct point3D *POI, struct point3D *normal, int col){
	double value = 0;
	//Return if at the end of the lightsource list
	if(lightSource == NULL){
		return 0;
	}
	//Assign light source and point to be point light source by default
	double lightSourceR = lightSource->col.R;
	double lightSourceG = lightSource->col.G;
	double lightSourceB = lightSource->col.B;
	struct point3D lightVector = lightSource->p0;
	if(lightSource->isObj){
		//Change the values if it is area light source
		lightSourceR = lightSource->obj.col.R;
		lightSourceG = lightSource->obj.col.G;
		lightSourceB = lightSource->obj.col.B;
		double x, y, z;
		double accx, accy, accz = 0;
		for(int i = 0; i < AREALIGHTSAMPLES; i++){
			lightSource->obj.randomPoint(&lightSource->obj, &x, &y, &z);
			accx += x;
			accy += y;
			accz += z;
		}
		accx = accx/AREALIGHTSAMPLES;
		accy = accy/AREALIGHTSAMPLES;
		accz = accz/AREALIGHTSAMPLES;
		lightVector = {.px = accx, .py = accy, .pz = accz, .pw= 0};
	}
	subVectors(POI, &lightVector);
	normalize(&lightVector);
	normalize(normal);
	if(col == 1){
		value = dot(normal, &lightVector) * lightSourceR;
	} else if(col == 2){
		value = dot(normal, &lightVector) * lightSourceG;		
	} else {
		value = dot(normal, &lightVector) * lightSourceB;
	}
	return value + lightDots(lightSource->next, collisionObject, POI, normal, col);
}

//Iteratively goes through each light sources and calculates the light value for each at a point.
//For col, 1 = Red, 2 = Green, 3 = Blue
double specLights(struct pointLS *lightSource, struct object3D *collisionObject, struct point3D *POI, struct point3D *normal, int col, struct point3D *camCoords){
	double value = 0;
	//Return if at the end of the lightsource list
	if(lightSource == NULL){
		return 0;
	}
	//Assign light source and point to be point light source by default
	double lightSourceR = lightSource->col.R;
	double lightSourceG = lightSource->col.G;
	double lightSourceB = lightSource->col.B;
	struct point3D sourceOfLight = lightSource->p0;
	//If the light source is an object take an average of multiple possible rays
	if(lightSource->isObj){
		//Change the values if it is area light source
		lightSourceR = lightSource->obj.col.R;
		lightSourceG = lightSource->obj.col.G;
		lightSourceB = lightSource->obj.col.B;
		double x = 0, y = 0, z = 0;
		double accx = 0, accy = 0, accz = 0;
		for(int i = 0; i < AREALIGHTSAMPLES; i++){
			lightSource->obj.randomPoint(&lightSource->obj, &x, &y, &z);
			accx += x;
			accy += y;
			accz += z;
		}
		//printf("lightVector:(%f,%f,%f)\n", accx, accy, accz);
		accx = accx/AREALIGHTSAMPLES;
		accy = accy/AREALIGHTSAMPLES;
		accz = accz/AREALIGHTSAMPLES;
		//lightVector = {.px = accx, .py = accy, .pz = accz, .pw= 0};
		//printmatrix(lightSource->obj.T);
		sourceOfLight = {.px = accx, .py = accy, .pz = accz, .pw= 1};
		//printf("dot:%f\n", dot(&lightVector, &lightVector));
	}
	//printf("LightVector:(%f,%f,%f)\n", lightVector.px, lightVector.py, lightVector.pz);
	struct point3D lightVector = sourceOfLight;
	subVectors(POI, &lightVector);
	//normalize(&lightVector);
	normalize(normal);
	if(dot(normal, &lightVector) >= 0){
		//Ray from lightsource to object at POI, this is d vector
		lightVector = *POI;
		//Direction is POI - lightSource
		subVectors(&sourceOfLight, &lightVector);
		//Reflected ray r is given by d - 2(d*n)n
		normalize(normal);
		double d2n = 2 * dot(&lightVector, normal);
		struct point3D adjNormal = {.px = normal->px*d2n, .py = normal->py*d2n, .pz = normal->pz*d2n};
		//The reflected vector R for phong is now stored in lightVector
		subVectors(&adjNormal, &lightVector);
		normalize(&lightVector);
		//printf("LightVector:(%f,%f,%f)\n", lightVector.px, lightVector.py, lightVector.pz);
		if(col == 1){
			value = pow(dot(camCoords, &lightVector),collisionObject->shinyness) * lightSourceR;
		} else if(col == 2){
			value = pow(dot(camCoords, &lightVector),collisionObject->shinyness) * lightSourceG;
		} else {
			value = pow(dot(camCoords, &lightVector),collisionObject->shinyness) * lightSourceB;
		}
	}
	
	//printf("Value: %f", value);
	if(value < 0){
		value = 0;
	}
	
	return value + specLights(lightSource->next, collisionObject, POI, normal, col, camCoords);
}

//Helper function for findFirstHit
//Gets the smallest non-zero intersection lambda point for ray and all objects in the scene
//by recursing through the object linked list
void recurseObjectList(struct object3D *origination, struct object3D *item, double *lambda, struct ray3D *ray, 
	struct point3D *p, struct point3D *n, double *a, double *b, struct object3D **intersectObject, int ignoreLightSource){
	if(item == NULL){//Exit if at end of object list
		return;
	} else if(item == origination){//Ignore the object if the ray originated from the object
		//printf("True\n");
		recurseObjectList(origination, item->next, lambda, ray, p, n, a, b, intersectObject, ignoreLightSource);
	} else {
		double testLambda = -1;
		struct point3D testP;
		struct point3D testN;
		double testA = 0;
		double testB = 0;
		item->intersect(item, ray, &testLambda, &testP, &testN, &testA, &testB);
		//printf("TestLambda:%f\n", testLambda);
		//printf(" %f,", testLambda);
		if((testLambda >= 0) && (testLambda < *lambda || *lambda <= 0)){//Replace the current values if lambda is lower
			if(!(ignoreLightSource && item->isLightSource)){//Do not replace values if you ignore light sources and the collided object is a light source
				*lambda = testLambda;
				*p = testP;
				*n = testN;
				*a = testA;
				*b = testB;
				*intersectObject = item;
			}
		}
		recurseObjectList(origination, item->next, lambda, ray, p, n, a, b, intersectObject, ignoreLightSource);
	}
}

void findFirstHit(struct ray3D *ray, double *lambda, struct object3D *Os, struct object3D **obj, struct point3D *p, struct point3D *n, double *a, double *b, int ignoreLightSource)
{
	
 // Find the closest intersection between the ray and any objects in the scene.
 // Inputs:
 //   *ray    -  A pointer to the ray being traced
 //   *Os     -  'Object source' is a pointer toward the object from which the ray originates. It is used for reflected or refracted rays
 //              so that you can check for and ignore self-intersections as needed. It is NULL for rays originating at the center of
 //              projection
 // Outputs:
 //   *lambda -  A pointer toward a double variable 'lambda' used to return the lambda at the intersection point
 //   **obj   -  A pointer toward an (object3D *) variable so you can return a pointer to the object that has the closest intersection with
 //              this ray (this is required so you can do the shading)
 //   *p      -  A pointer to a 3D point structure so you can store the coordinates of the intersection point
 //   *n      -  A pointer to a 3D point structure so you can return the normal at the intersection point
 //   *a, *b  -  Pointers toward double variables so you can return the texture coordinates a,b at the intersection point

 /////////////////////////////////////////////////////////////
 // TO DO: Implement this function. See the notes for
 // reference of what to do in here
 /////////////////////////////////////////////////////////////
 //printf("Ray:");
 recurseObjectList(Os, object_list, lambda, ray, p, n, a, b, obj, ignoreLightSource);
 //printf("\n");
}

//Bounds the value to a range of [min, max]
void bound(double *value, double max, double min){
	if(*value > max){
		*value = max;
	}
	if(*value < min){
		*value = min;
	}
}

void rtShade(struct object3D *obj, struct point3D *p, struct point3D *n, struct ray3D *ray, int depth, double a, double b, struct colourRGB *col, struct objStack *	position)
{
 // This function implements the shading model as described in lecture. It takes
 // - A pointer to the first object intersected by the ray (to get the colour properties)
 // - The coordinates of the intersection point (in world coordinates)
 // - The normal at the point
 // - The ray (needed to determine the reflection direction to use for the global component, as well as for
 //   the Phong specular component)
 // - The current racursion depth
 // - The (a,b) texture coordinates (meaningless unless texture is enabled)
 //
 // Returns:
 // - The colour for this ray (using the col pointer)
 //

 double R,G,B;			// Colour for the object in R G and B

 if (obj->texImg==NULL)		// Not textured, use object colour
 {
  R=obj->col.R;
  G=obj->col.G;
  B=obj->col.B;
 }
 else
 {
  // Get object colour from the texture given the texture coordinates (a,b), and the texturing function
  // for the object. Note that we will use textures also for Photon Mapping.
  obj->textureMap(obj->texImg,a,b,&R,&G,&B);
 }

 //////////////////////////////////////////////////////////////
 // TO DO: Implement this function. Refer to the notes for
 // details about the shading model.
 //////////////////////////////////////////////////////////////
 //Calculates if a shadow exists
 double shadowConstant = shadowChecker(obj, p, n, light_list);
 //if(shadowConstant < 1 && shadowConstant > 0){
	//printf("%f\n", shadowConstant);
 // }
 
 struct colourRGB ambient = {.R = R, .G = G, .B = B}; 
 struct colourRGB diffuse = {.R = lightDots(light_list, obj, p, n, 1), 
							.G = lightDots(light_list, obj, p, n, 2),
							.B = lightDots(light_list, obj, p, n, 3)}; 
 struct point3D camCoord = ray->d;
 //Invert the direction of the ray to get the direction from the POI to the camera eye
 scalarMultVector(&camCoord, -1);
 struct colourRGB specular = {.R = specLights(light_list, obj, p, n, 1, &camCoord), 
								.G = specLights(light_list, obj, p, n, 2, &camCoord), 
								.B = specLights(light_list, obj, p, n, 3, &camCoord)};
 struct colourRGB Ispec = {.R = 0, .G = 0, .B = 0};
 struct colourRGB Irefr = {.R = 0, .G = 0, .B = 0};

 
 if(depth <= MAX_DEPTH){
	//Check to see if the object has global specular reflection
	if(obj->alb.rg > 0){
		struct point3D dn2 = *n;
		struct point3D reflected = ray->d;
		// d - 2(d*n)n
		scalarMultVector(&dn2, 2*dot(&ray->d, &dn2));
		subVectors(&dn2, &reflected);
		reflected.pw = 0;
		//The reflected ray direction after hitting the normal is stored in reflected
		struct ray3D refRay;
		//Initializes the new ray
		initRay(&refRay, p, &reflected);
		double lambda2 = -1;
		double a2,b2;
		struct object3D *obj2 = object_list;
		struct point3D p2;
		struct point3D n2;
		//printf("(%f,%f,%f)+(%f,%f,%f,%f)\n", refRay.p0.px, refRay.p0.py, refRay.p0.pz, refRay.d.px, refRay.d.py, refRay.d.pz, refRay.d.pw);
		//printf("Reflection\n");
		rayTrace(&refRay, depth+1, &Ispec, obj, position);
	}
	
	//Implement refraction if object is not opaque and the refractive component is not 0
	if(obj->alb.rt > 0 && obj->alpha < 1){
		//printf("(%f,%f,%f)+(%f,%f,%f)\n", ray->p0.px, ray->p0.py, ray->p0.pz, ray->d.px, ray->d.py, ray->d.pz);
		double c = dot(n,&ray->d);
		double n1 = peek(position);
		double n2 = 0;
		if(c < 0){
			//printf("in\n");
			n2 = obj->r_index;
			push(position, n2);
		} else{
			//printf("out\n");
			pop(position);
			n2 = peek(position);
		}
		//printf("Indexes: n1=%f, n2=%f", n1, n2);
		//printf("Normal: (%f,%f,%f)\n", n->px, n->py, n->pz);
		//Refracted ray direction = r*ray->d + (rc - n*sqrt(1 - (r^2)*(1 - (c^2))))n
		double r = n1/n2;
		double directionConstant = r*c - sqrt(1 - r*r*(1 - c*c));
		struct point3D b = ray->d;
		struct point3D nom = *n;
		scalarMultVector(&nom, directionConstant);
		scalarMultVector(&b, r);
		addVectors(&nom, &b);
		b.pw = 0;
		struct ray3D refractedRay;
		initRay(&refractedRay, p, &b);
		rayTrace(&refractedRay, depth+1, &Irefr, NULL, position);
		if(c < 0){
			pop(position);
		}
	}
 }
 
 // Be sure to update 'col' with the final colour computed here!
 //Change number to change what components the output uses. Refer to colorCalc to see values and outcomes
 colorCalc(col, &ambient, &diffuse, &specular, shadowConstant, &Ispec, &Irefr, n, obj, 8);
 return;

}

void rayTrace(struct ray3D *ray, int depth, struct colourRGB *col, struct object3D *Os, struct objStack *position)
{
 // Trace one ray through the scene.
 //
 // Parameters:
 //   *ray   -  A pointer to the ray being traced
 //   depth  -  Current recursion depth for recursive raytracing
 //   *col   - Pointer to an RGB colour structure so you can return the object colour
 //            at the intersection point of this ray with the closest scene object.
 //   *Os    - 'Object source' is a pointer to the object from which the ray 
 //            originates so you can discard self-intersections due to numerical
 //            errors. NULL for rays originating from the center of projection. 
 
 double lambda = -1;		// Lambda at intersection
 double a,b;		// Texture coordinates
 //Have to initialize with a default value so seg faults do not occur
 struct object3D *obj = object_list;	// Pointer to object at intersection
 struct point3D p;	// Intersection point
 struct point3D n;	// Normal at intersection
 struct colourRGB I;	// Colour returned by shading function

 if (depth>MAX_DEPTH)	// Max recursion depth reached. Return invalid colour.
 {
  col->R=-1;
  col->G=-1;
  col->B=-1;
  return;
 }

 col->R=0;
 col->G=0;
 col->B=0;
 
 ///////////////////////////////////////////////////////
 // TO DO: Complete this function. Refer to the notes
 // if you are unsure what to do here.
 ///////////////////////////////////////////////////////

 findFirstHit(ray, &lambda, Os, &obj, &p, &n, &a, &b, 0);
 //printf("Depth:%d, Lambda:%f\n", depth, lambda);
 if(lambda > 0){
	rtShade(obj, &p, &n, ray, depth, a, b, col, position);
	bound(&col->R, 1, 0);
	bound(&col->G, 1, 0);
	bound(&col->B, 1, 0);
	
 } else{
	//Ray collides with nothing, return background
	//Assume the background is pitch black
	col->R=0;
	col->G=0;
	col->B=0;
 }
}

void printMatrices(struct object3D *o_list){
	if(o_list == NULL){
		return;
	}
	printf("Matrix(%f, %f, %f) and its inverse:\n", o_list->col.R, o_list->col.G, o_list->col.B);
	printmatrix(o_list->T);
	printmatrix(o_list->Tinv);
	printMatrices(o_list->next);
}

struct multiThreadInput{
	struct view *cam;
	struct image *im;
	int antialiasing;
	unsigned char *rgbIm;
	double du;
	double dv;
	int threadNumber;
	int sx;
	struct point3D e;
};

void *computeRay(void *input){
	struct multiThreadInput *args = (multiThreadInput *)input;
	for(int x = args->threadNumber ; x< args->sx*args->sx; x = x + NUM_THREADS){
		int i = x%args->sx;
		int j = x/args->sx;
		struct colourRGB col;
		if(!args->antialiasing){
			struct point3D coords = {.px = args->cam->wl + i*args->du, .py = args->cam->wt + j*args->dv, .pz = args->cam->f, .pw = 1};
	
			//Turn camera coordinates into world coordinates for the pixel
			matVecMult(args->cam->C2W, &coords);
	
			//Get the direction of the ray by subtracting the pixel coords by camera in world coords
			struct point3D directs = coords;
	
			//a = e, b = directs, directs = directs - e
			subVectors(&args->e, &directs);
			normalize(&directs);
			struct ray3D pixelRay;
			initRay(&pixelRay, &coords, &directs);
			//Set direction pw to 0 to prevent matrix multiplication of inverse transformation from transforming the direction matrix
			pixelRay.d.pw = 0;
			rayTrace(&pixelRay, 1, &col, NULL, NULL);
		} else{
			struct colourRGB colACC = {.R = 0, .G = 0, .B = 0};
			for(int k = 0; k < AANumber; k++){
				struct colourRGB colTemp;
				double randx = (double)rand()/(double)RAND_MAX;
				double randy = (double)rand()/(double)RAND_MAX;
				struct point3D coords = {.px = args->cam->wl +(i+randx)*args->du, .py = args->cam->wt + (j+randy)*args->dv, .pz = args->cam->f, .pw = 1};
				//Turn camera coordinates into world coordinates for the pixel
				matVecMult(args->cam->C2W, &coords);
	
				//Get the direction of the ray by subtracting the pixel coords by camera in world coords
				struct point3D directs = coords;
	
				//a = e, b = directs, directs = directs - e
				subVectors(&args->e, &directs);
				normalize(&directs);
				struct ray3D pixelRay;
				initRay(&pixelRay, &coords, &directs);
				//Set direction pw to 0 to prevent matrix multiplication of inverse transformation from transforming the direction matrix
				pixelRay.d.pw = 0;
				rayTrace(&pixelRay, 1, &colTemp, NULL, NULL);
				colACC.R += colTemp.R;
				colACC.G += colTemp.G;
				colACC.B += colTemp.B;
			}
			col.R = colACC.R/AANumber;
			col.G = colACC.G/AANumber;
			col.B = colACC.B/AANumber;
		}
		//printf("Write_%d\n", args->j*args->sx + args->i);
		(*(args->rgbIm+((i+(j*args->sx))*3)+0))=(int)(255*col.R);
		(*(args->rgbIm+((i+(j*args->sx))*3)+1))=(int)(255*col.G);
		(*(args->rgbIm+((i+(j*args->sx))*3)+2))=(int)(255*col.B);
	}
	//printf("Done_%d\n", args->j*args->sx + args->i);
	pthread_exit(NULL);
	return NULL;
}

int main(int argc, char *argv[])
{
 // Main function for the raytracer. Parses input parameters,
 // sets up the initial blank image, and calls the functions
 // that set up the scene and do the raytracing.
 struct image *im;	// Will hold the raytraced image
 struct view *cam;	// Camera and view for this scene
 int sx;		// Size of the raytraced image
 int antialiasing;	// Flag to determine whether antialiaing is enabled or disabled
 int multithreading; // Flah to determine whether multithreading is enabled or disabled
 struct point3D e;		// Camera view parameters 'e', 'g', and 'up'
 struct point3D g;
 struct point3D up;
 char output_name[1024];	// Name of the output file for the raytraced .ppm image
 double du, dv;			// Increase along u and v directions for pixel coordinates
 struct point3D pc,d;		// Point structures to keep the coordinates of a pixel and
				// the direction or a ray
 struct ray3D ray;		// Structure to keep the ray from e to a pixel
 struct colourRGB col;		// Return colour for raytraced pixels
 struct colourRGB background;   // Background colour
 int i,j;			// Counters for pixel coordinates
 unsigned char *rgbIm;

 clock_gettime(CLOCK_MONOTONIC, &start);

 if (argc<6)
 {
  fprintf(stderr,"RayTracer: Can not parse input parameters\n");
  fprintf(stderr,"USAGE: RayTracer size rec_depth antialias output_name\n");
  fprintf(stderr,"   size = Image size (both along x and y)\n");
  fprintf(stderr,"   rec_depth = Recursion depth\n");
  fprintf(stderr,"   antialias = A single digit, 0 disables antialiasing. Anything else enables antialiasing\n");
  fprintf(stderr,"   multithreading = A single digit, 0 disables multithreading. Anything else enables multithreading\n");
  fprintf(stderr,"   output_name = Name of the output file, e.g. MyRender.ppm\n");
  exit(0);
}
 sx=atoi(argv[1]);
 MAX_DEPTH=atoi(argv[2]);
 if (atoi(argv[3])==0) antialiasing=0; else antialiasing=1;
 if (atoi(argv[4])==0) multithreading=0; else multithreading=1;
 strcpy(&output_name[0],argv[5]);

 fprintf(stderr,"Rendering image at %d x %d\n",sx,sx);
 fprintf(stderr,"Recursion depth = %d\n",MAX_DEPTH);
 if (!antialiasing) fprintf(stderr,"Antialising is off\n");
 else fprintf(stderr,"Antialising is on\n");
 if (!multithreading) fprintf(stderr,"Multithreading is off\n");
 else fprintf(stderr,"Multithreading is on\n");
 fprintf(stderr,"Output file name: %s\n",output_name);

 object_list=NULL;
 light_list=NULL;
 texture_list=NULL;

 // Allocate memory for the new image
 im=newImage(sx, sx);
 if (!im)
 {
  fprintf(stderr,"Unable to allocate memory for raytraced image\n");
  exit(0);
 }
 else rgbIm=(unsigned char *)im->rgbdata;

 ///////////////////////////////////////////////////
 // TO DO: You will need to implement several of the
 //        functions below. For Assignment 2, you can use
 //        the simple scene already provided. But
 //        for Assignment 3 you need to create your own
 //        *interesting* scene.
 ///////////////////////////////////////////////////
 buildScene();		// Create a scene. This defines all the
			// objects in the world of the raytracer
 //////////////////////////////////////////
 // TO DO: For Assignment 2 you can use the setup
 //        already provided here. For Assignment 3
 //        you may want to move the camera
 //        and change the view parameters
 //        to suit your scene.
 //////////////////////////////////////////
 // Mind the homogeneous coordinate w of all vectors below. DO NOT
 // forget to set it to 1, or you'll get junk out of the
 // geometric transformations later on.

 // Camera center is at (0,0,-1)
 e.px=0;
 e.py=0;
 e.pz=-1;
 e.pw=1;

 // To define the gaze vector, we choose a point 'pc' in the scene that
 // the camera is looking at, and do the vector subtraction pc-e.
 // Here we set up the camera to be looking at the origin.
 g.px=0-e.px;
 g.py=0-e.py;
 g.pz=0-e.pz;
 g.pw=1;
 // In this case, the camera is looking along the world Z axis, so
 // vector w should end up being [0, 0, -1]

 // Define the 'up' vector to be the Y axis
 up.px=0;
 up.py=1;
 up.pz=0;
 up.pw=1;

 // Set up view with given the above vectors, a 4x4 window,
 // and a focal length of -1 (why? where is the image plane?)
 // Note that the top-left corner of the window is at (-2, 2)
 // in camera coordinates.
 cam=setupView(&e, &g, &up, -1, -2, 2, 4);

 if (cam==NULL)
 {
  fprintf(stderr,"Unable to set up the view and camera parameters. Our of memory!\n");
  cleanup(object_list,light_list,texture_list);
  deleteImage(im);
  exit(0);
 }

 // Set up background colour here
 background.R=0;
 background.G=0;
 background.B=0;

 // Do the raytracing
 //////////////////////////////////////////////////////
 // TO DO: You will need code here to do the raytracing
 //        for each pixel in the image. Refer to the
 //        lecture notes, in particular, to the
 //        raytracing pseudocode, for details on what
 //        to do here. Make sure you undersand the
 //        overall procedure of raytracing for a single
 //        pixel.
 //////////////////////////////////////////////////////
 du=cam->wsize/(sx-1);		// du and dv. In the notes in terms of wl and wr, wt and wb,
 dv=-cam->wsize/(sx-1);		// here we use wl, wt, and wsize. du=dv since the image is
				// and dv is negative since y increases downward in pixel
				// coordinates and upward in camera coordinates.

 //fprintf(stderr,"View parameters:\n");
 //fprintf(stderr,"Left=%f, Top=%f, Width=%f, f=%f\n",cam->wl,cam->wt,cam->wsize,cam->f);
 //fprintf(stderr,"Camera to world conversion matrix (make sure it makes sense!):\n");
 //printmatrix(cam->C2W);
 //fprintf(stderr,"World to camera conversion matrix:\n");
 //printmatrix(cam->W2C);
 //fprintf(stderr,"\n");
 
 //Structure to test single intersection points for implicit surfaces
 //struct point3D direction = {.px = 0,.py = 0, .pz = 1, .pw = 0};
 //struct point3D point = { .px = 0, .py = 0, .pz = 1, .pw = 1};
 //struct ray3D rayz;
 //initRay(&rayz, &point, &direction);
 //double l, as, bs;
 //struct object3D *object = object_list;
 //struct point3D pont, norm;
 //implicitIntersect(object, &rayz, &l, &pont, &norm, &as, &bs);
 //printf("Lambda:%f\n", l);
 
 if(multithreading){
	pthread_t somethreads[NUM_THREADS];
	struct multiThreadInput mtargs[NUM_THREADS];
	//fprintf(stderr,"Rendering row: ");
	for(int i = 0; i < NUM_THREADS; i++){
		mtargs[i].cam = cam;
		mtargs[i].im = im;
		mtargs[i].antialiasing = antialiasing;
		mtargs[i].rgbIm = rgbIm;
		mtargs[i].du = du;
		mtargs[i].dv = dv;
		mtargs[i].e = e;
		mtargs[i].threadNumber = i;
		mtargs[i].sx = sx;
		if(pthread_create(&somethreads[i], NULL, &computeRay, (void *)&mtargs[i]) != 0){ // Create threads for 1 row of the image
			printf("Bad news, you need to be able to initialize %d consecutive threads at a time", sx);
			break;
		}
	}
	for (int j=0;j<NUM_THREADS;j++)		// For each of the pixels in the image
	{
		pthread_join(somethreads[j], NULL);
	} // end for j
 } else{
	for(int j = 0; j < sx; j++){
		for(int i= 0; i< sx; i++){
			if(!antialiasing){
			struct point3D coords = {.px = cam->wl + i*du, .py = cam->wt + j*dv, .pz = cam->f, .pw = 1};
	
			//Turn camera coordinates into world coordinates for the pixel
			matVecMult(cam->C2W, &coords);
	
			//Get the direction of the ray by subtracting the pixel coords by camera in world coords
			struct point3D directs = coords;
	
			//a = e, b = directs, directs = directs - e
			subVectors(&e, &directs);
			normalize(&directs);
			struct ray3D pixelRay;
			initRay(&pixelRay, &coords, &directs);
			//Set direction pw to 0 to prevent matrix multiplication of inverse transformation from transforming the direction matrix
			pixelRay.d.pw = 0;
			rayTrace(&pixelRay, 1, &col, NULL, NULL);
		} else{
			struct colourRGB colACC = {.R = 0, .G = 0, .B = 0};
			for(int k = 0; k < AANumber; k++){
				struct colourRGB colTemp;
				double randx = (double)rand()/(double)RAND_MAX;
				double randy = (double)rand()/(double)RAND_MAX;
				struct point3D coords = {.px = cam->wl +(i+randx)*du, .py = cam->wt + (j+randy)*dv, .pz = cam->f, .pw = 1};
				//Turn camera coordinates into world coordinates for the pixel
				matVecMult(cam->C2W, &coords);
	
				//Get the direction of the ray by subtracting the pixel coords by camera in world coords
				struct point3D directs = coords;
	
				//a = e, b = directs, directs = directs - e
				subVectors(&e, &directs);
				normalize(&directs);
				struct ray3D pixelRay;
				initRay(&pixelRay, &coords, &directs);
				//Set direction pw to 0 to prevent matrix multiplication of inverse transformation from transforming the direction matrix
				pixelRay.d.pw = 0;
				if(j == 512 && i == 461){
					printf("Ray:(%f,%f,%f)(%f,%f,%f)\n", pixelRay.p0.px, pixelRay.p0.py, pixelRay.p0.pz, pixelRay.d.px, pixelRay.d.py,pixelRay.d.pz);
				}
				rayTrace(&pixelRay, 1, &colTemp, NULL, NULL);
				colACC.R += colTemp.R;
				colACC.G += colTemp.G;
				colACC.B += colTemp.B;
			}
			col.R = colACC.R/AANumber;
			col.G = colACC.G/AANumber;
			col.B = colACC.B/AANumber;
		}
			//printf("R: %f,G: %f,B: %f\n", col.R, col.G, col.B);
			(*(rgbIm+((i+(j*sx))*3)+0))=(int)(255*col.R);
			(*(rgbIm+((i+(j*sx))*3)+1))=(int)(255*col.G);
			(*(rgbIm+((i+(j*sx))*3)+2))=(int)(255*col.B);
			//printf("Row %d Col %d Done\n", j, i);
		}
			printf("Row %d Done\n", j);
	}
 }
	
	clock_gettime(CLOCK_MONOTONIC, &end);
	elapsed = (end.tv_sec - start.tv_sec);
	printf("Job finished in %f seconds\n", elapsed);

	// Output rendered image
	imageOutput(im,output_name);

	// Exit section. Clean up and return.
	cleanup(object_list,light_list,texture_list);		// Object, light, and texture lists
	deleteImage(im);					// Rendered image
	free(cam);						// camera view
	exit(0);
}

