 // Sets up all objects in the scene. This involves creating each object,
 // defining the transformations needed to shape and position it as
 // desired, specifying the reflectance properties (albedos and colours)
 // and setting up textures where needed.
 // Light sources must be defined, positioned, and their colour defined.
 // All objects must be inserted in the object_list. All light sources
 // must be inserted in the light_list.
 //
 // To create hierarchical objects:
 //    You must keep track of transformations carried out by parent objects
 //    as you move through the hierarchy. Declare and manipulate your own
 //    transformation matrices (use the provided functions in utils.c to
 //    compound transformations on these matrices). When declaring a new
 //    object within the hierarchy
 //    - Initialize the object
 //    - Apply any object-level transforms to shape/rotate/resize/move
 //      the object using regular object transformation functions
 //    - Apply the transformations passed on from the parent object
 //      by pre-multiplying the matrix containing the parent's transforms
 //      with the object's own transformation matrix.
 //    - Compute and store the object's inverse transform as usual.
 //
 // NOTE: After setting up the transformations for each object, don't
 //       forget to set up the inverse transform matrix!

 struct object3D *o;
 struct pointLS *l;
 struct point3D p;

 o=newSphere(.2,.95,.75,.75,1,.75,.95,.55,1,1,6);
 Scale(o,1.5,1,1.5);
 RotateX(o, PI/2);
 Translate(o,-2.2,1.75,2.35);
 invert(&o->T[0][0],&o->Tinv[0][0]);
 loadTexture(o,"./Texture/bump1.ppm",2,&texture_list);
 insertObject(o,&object_list);

 //Cylinder just for testing purposes
 o=newCyl(.2,.95,.75,.75,.3,.75,.95,.55,1,1,6);
 Scale(o,2,2,4);
 RotateX(o, PI/2);
 RotateY(o, PI/2);
 Translate(o,0,2,6);
 invert(&o->T[0][0],&o->Tinv[0][0]);
 loadTexture(o, "./Texture/fourSquare.ppm", 1, &texture_list);
 loadTexture(o, "./Texture/normalbrick.ppm", 2, &texture_list);
 insertObject(o,&object_list);

 o=newPlane(.2,.75,.25,.05,.3,.55,.8,.75,1,1,2);
 Scale(o,11,11,11);
 RotateZ(o,PI/4);
 RotateX(o,PI/2);
 Translate(o,0,-4,5);
 invert(&o->T[0][0],&o->Tinv[0][0]);
 loadTexture(o,"./Texture/normalbrick.ppm",2,&texture_list);
 insertObject(o,&object_list);
 
 o=newPlane(.2,.75,.25,.05,.3,1,.1,.1,1,1,2);
 Scale(o,100,100,100);
 RotateY(o,-PI/4);
 Translate(o,-7,7,13);
 invert(&o->T[0][0],&o->Tinv[0][0]);
 loadTexture(o,"./Texture/bump1.ppm",2,&texture_list);
 insertObject(o,&object_list);
 
 o=newPlane(.2,.75,.25,.05,.3,.1,1,.1,1,1,2);
 Scale(o,100,100,100);
 RotateY(o,PI/4);
 Translate(o,7,7,13);
 invert(&o->T[0][0],&o->Tinv[0][0]);
 loadTexture(o,"./Texture/normalbrick.ppm",2,&texture_list);
 insertObject(o,&object_list);
 
 o=newSphere(1,0,0,0,0,1,1,1,1,1,1);
 Scale(o, 5, 5, 5);
 Translate(o, 0, 25.5, -3.5);
 invert(&o->T[0][0], &o->Tinv[0][0]);
 o->isLightSource = 1;
 insertObject(o, &object_list);
 l=newALS(o);
 insertPLS(l, &light_list);