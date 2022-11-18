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

 // Simple scene for Assignment 3:
 // Insert a couple of objects. A plane and two spheres
 // with some transformations.

 // Note the parameters: ra, rd, rs, rg, R, G, B, alpha, r_index, and shinyness)

 o=newSphere(.2,.95,.35,.35,.3,1,.25,.25,1,1.52,6);		// Initialize a sphere
 Scale(o,1.5,.75,.75);					// Apply a few transforms (Translate * Rotate * Scale)
 RotateZ(o,PI/4);					
 Translate(o,2.0,2.5,1.5);
 invert(&o->T[0][0],&o->Tinv[0][0]);			// Compute the inverse transform * DON'T FORGET TO DO THIS! *

 // If needed, this is how you load a texture map
 // loadTexture(o,"./Texture/mosaic2.ppm",1,&texture_list);	// This loads a texture called 'mosaic2.ppm'. The
								// texture gets added to the texture list, and a
								// pointer to it is stored within this object in the
								// corresponding place. The '1' indicates this image
								// will be used as a texture map. Use '2' to load
								// an image as a normal map, and '3' to load an
								// alpha map. Texture and normal maps are RGB .ppm
								// files, alpha maps are grayscale .pgm files.
								// * DO NOT * try to free image data loaded in this
								// way, the cleanup function already provided will do
								// this at the end.
 
 //loadTexture(o,"texture.ppm",1,&texture_list);
 //insertObject(o,&object_list);	// <-- If you don't insert the object into the object list,
								// nothing happens! your object won't be rendered.

 // That's it for defining a single sphere... let's add a couple more objects
 o=newSphere(.2,.95,.75,.75,0.5,.75,.95,.55,1,1,6);
 Scale(o,.75,.75,.75);
 Translate(o,-2.2,4.75,2.35);
 invert(&o->T[0][0],&o->Tinv[0][0]);
 //loadTexture(o,"./Texture/fourSquare.ppm",1,&texture_list);
 loadTexture(o,"./Texture/normalbrick.ppm",2,&texture_list);
 insertObject(o,&object_list);

 o=newSphere(.2,.95,.75,.75,0.75,.3,.5,.75,1,1,6);
 Scale(o,.75,.75,.75);
 Translate(o,2.2,4.75,2.35);
 invert(&o->T[0][0],&o->Tinv[0][0]);
 loadTexture(o,"./Texture/sky.ppm",1,&texture_list);
 insertObject(o,&object_list);

 o=newSphere(.2,.95,.75,.75,0.75,.35,.75,.35,0.2,1.1,6);
 Translate(o,1.4,1.15,1.35);
 invert(&o->T[0][0],&o->Tinv[0][0]);
 insertObject(o,&object_list);
 
 o=newSphere(.9,.95,.75,.75,0.75,.35,.75,.35,1,1,6);
 Scale(o, 0.25, 0.25, 0.25);
 Translate(o,1.4,1.15,1.35);
 invert(&o->T[0][0],&o->Tinv[0][0]);
 loadTexture(o,"./Texture/sky.ppm",1,&texture_list);
 insertObject(o,&object_list);
 
 o=newSphere(.2,.95,.75,.75,0.75,.95,.15,.55,0.2,1.1,6);
 Translate(o,-1.4,1.15,1.35);
 invert(&o->T[0][0],&o->Tinv[0][0]);
 insertObject(o,&object_list);
 
 o=newSphere(.9,.95,.75,.75,0.75,.95,.15,.55,1,1,6);
 Scale(o, 0.25, 0.25, 0.25);
 Translate(o,-1.4,1.15,1.35);
 invert(&o->T[0][0],&o->Tinv[0][0]);
 loadTexture(o,"./Texture/sky.ppm",1,&texture_list);
 insertObject(o,&object_list);
 
 o=newSphere(.2,.95,.75,.75,0.75,.95,.15,.55,0.1,1.52,6);
 Scale(o, 0.55, 0.55, 0.55);
 Translate(o,0,0.05,1.35);
 invert(&o->T[0][0],&o->Tinv[0][0]);
 insertObject(o,&object_list);
 
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
 loadTexture(o, "./Texture/wood.ppm", 1, &texture_list);
 insertObject(o,&object_list);
 
 o=newPlane(.2,.75,.25,.05,.3,1,.1,.1,1,1,2);
 Scale(o,11,11,11);
 RotateY(o,-PI/4);
 Translate(o,-7,7,13);
 invert(&o->T[0][0],&o->Tinv[0][0]);
 loadTexture(o, "./Texture/bump1.ppm", 1, &texture_list); 
 insertObject(o,&object_list);
 
 o=newPlane(.2,.75,.25,.05,.3,.1,1,.1,1,1,2);
 Scale(o,11,11,11);
 RotateY(o,PI/4);
 Translate(o,7,7,13);
 invert(&o->T[0][0],&o->Tinv[0][0]);
 loadTexture(o, "./Texture/bump1.ppm", 2, &texture_list);
 insertObject(o,&object_list);

 o=newImplicit(&implicitChubbs, &implicitChubbsNormal, .2,.95,.75,.75,.3,.75,.95,.55,1,1,6);
 Scale(o,.95,1.65,.65);
 RotateZ(o,-PI/1.5);
 RotateX(o, PI);
 Translate(o,0,3,2.35);
 invert(&o->T[0][0],&o->Tinv[0][0]);
 //insertObject(o,&object_list);
 //Implicit objects cannot have textures yet

 o=newImplicit(&implicitTangleCube, &implicitTangleCubeNormal,.2,.95,.75,.75,.3,.75,.95,.55,1,1,6);
 Scale(o,2,2,2);
 RotateY(o, PI/4);
 RotateX(o, PI/4);
 Translate(o, 2, 1, 8);
 invert(&o->T[0][0],&o->Tinv[0][0]);
 //insertObject(o,&object_list);

 //Area Light Source
 o=newSphere(1,0,0,0,0,1,1,1,1,1,1);
 Scale(o, 5, 5, 5);
 Translate(o, 0, 25.5, -3.5);
 invert(&o->T[0][0], &o->Tinv[0][0]);
 o->isLightSource = 1;
 insertObject(o, &object_list);
 l=newALS(o);
 insertPLS(l, &light_list);

 // End of simple scene for Assignment 2
 // Keep in mind that you can define new types of objects such as cylinders and parametric surfaces,
 // or, you can create code to handle arbitrary triangles and then define objects as surface meshes.
 //
 // Remember: A lot of the quality of your scene will depend on how much care you have put into defining
 //           the relflectance properties of your objects, and the number and type of light sources
 //           in the scene.
 
 ///////////////////////////////////////////////////////////////////////////////////////////////////////////
 // TO DO: For Assignment 3 you *MUST* define your own cool scene.
 //	   We will be looking for the quality of your scene setup, the use of hierarchical or composite
 //	   objects that are more interesting than the simple primitives from A2, the use of textures
 //        and other maps, illumination and illumination effects such as soft shadows, reflections and
 //        transparency, and the overall visual quality of your result. Put some work into thinking
 //        about these elements when designing your scene.
 ///////////////////////////////////////////////////////////////////////////////////////////////////////////
