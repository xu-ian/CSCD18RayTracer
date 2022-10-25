%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CSCD18 - Computer Graphics - UTSC
%%
%% Assignment 2 - Epicycloids (yeah! that's the name!)
%%
%% Read the handout carefully, and complete
%% the code required in the sections below.
%%
%% Script by F. Estrada, Oct. 2022
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% (1 mark)
%% TO DO: Complete the identification information
%%  below and sign the academic integrity statement.
%%
%% Identification information:
%%
%% Student 1:
%%
%%	Name: Ian Xu
%%	Student Number: 1006319208
%%	UTORid: xuian
%%
%% Student 2:
%%
%%	Name:
%%	Student Number:
%%	UTORid:
%%
%% We hereby confirm that the solution we provide for
%% this assignment is our own, and that it was developed
%% independently of our peers, without the use of 
%% unauthorized resources or help from peers or individuals
%% not in the course.
%%
%% Sign with your name:
%%
%%	Student 1: Ian Xu
%%
%%	Student 2:
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function []=epicycloid(r,k,cycles)

%% This will trace a hypotrochoid with the given constants
%% r, and k, with R=k*r (which control the shape of the curve)
%% 'cycles' controls how many cycles of the sine and cosine
%% waves will be plotted. 'cycles' only makes a difference
%% when r, R, and/or k are non-integer.

%% The actual parameter is t

t=[0:.01:cycles*2*pi];

R=k*r;
x=((R+r)*cos(t))-(r*cos(((R+r)/r)*t));
y=((R+r)*sin(t))-(r*sin(((R+r)/r)*t));
z=r*cos((R/r)*t);

s=sprintf('The Epicyloid, R=%d, r=%d, k=%d, cycles=%d',R,r,k,cycles);
figure(1);clf;axis equal;title(s);hold on;
%plot(x,y,'.-','color',[0 0 1]);
plot3(x,y,z,'.-','color',[0 0 1]);
view(3)
grid on

%% The plot above may look weird in 3D, use the GUI to
%% rotate the curve until you can see the 3D shape of the
%% parametric curve!

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% (2 marks)
%% Part 1: TO DO:
%%
%% Starting at 0, and progressing to cycles*2*pi at
%% intervals of .05, compute the tangent vector
%% for the curve using the equations you derived.
%%
%% Each tangent vector will be stored as a row
%% in an array for later use, and we'll
%% plot them on the curve so you can see if 
%% they are reasonable.
%% 
%% Complete the code below to obtain the tangent
%% vectors.
%%
%% IN THE SAME FOR LOOP, but commented out
%% until you have the tangent vectors: 
%%
%% TO DO:
%%
%% (4 marks)
%% Part 2:
%%
%% We want to turn the epicycloid curve into a funky 
%% tube-shape we will draw
%% a small circle, perpendicular to the curve, at 
%% each location on the curve. 
%%
%% We need to find the tangent plane at each point,
%% but all we have is a tangent vector.
%%
%% In the space below, complete code to obtain the
%% tangent plant at each point in the curve. Note
%% that we need 2 orthogonal, unit vectors on this
%% plane in order to be able to draw the circles.
%% 
%% These vectors will be stored in arrays called
%% u_vec, and v_vec. Much like the tangents 
%% array above.
%%
%% HINT: The tangent vector is orthogonal to these
%% two. Also, we solved a similar problem when
%% setting up a camera's coordinate frame vectors
%% u, v, and w!
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tangents=[];
h=.0001;
for i=0:.05:12*pi

  %% Determine the current point [xa, ya, za] on the hypotrochoid
  %% corresponding to t=i from the for loop above.

  xa=((R+r)*cos(i))-(r*cos(((R+r)/r)*i));     % Dummy point on curve, replace with actual
  ya=((R+r)*sin(i))-(r*sin(((R+r)/r)*i));     % location on the curve from the curve's
  za=r*cos((R/r)*i);     % parametric equations!

  %% Part 1 code goes just below these comments!
  %% Compute tx, ty, and tz, the components of
  %% the tangent vector

  %dx(t)/dt = d((R+r)cos(t))- d(r*cos((R+r)*t/r)
  %dx(t)/dt = -(R+r)sin(t) + (R+r)sin((R+r)*t/r)
  %dx(t)/dt = (R+r)(sin((R+r)*t/r) - sin(t))
  %dy(t)/dt = d((R+r)sin(t))- d(r*sin((R+r)*t/r)
  %dy(t)/dt = -(R+r)cos(t) + (R+r)cos((R+r)*t/r)
  %dy(t)/dt = (R+r)(cos((R+r)*t/r) - cos(t))
  %dz(t)/dt = -Rsin(R*t/r)

  tx=(R + r)*(sin(((R+r)/r)*i) - sin(i));
  ty=(R + r)*(cos(((R+r)/r)*i) - cos(i));         % DUMMY VECTOR, replace with correct one. 
  tz=-R*sin((R/r)*i);
  
  tangents(end+1,:)=[tx ty tz];

  line([xa xa+(.02*tx)],[ya ya+(.02*ty)],[za za+(.02*tz)],'color',[1 0 0],'linewidth',2.5);

  %% IF you computed the tangent vectors correctly, you should see little red lines that
  %% align with the blue curve, these cover a pair of curve points.
  %% If the lines do not align with the blue curve, something went wrong, 
  %% and you should have a second look at your computations above.

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %% SAVE THIS FIGURE, you need to submit it. And the code below will change the plot!
  %%
  %% For the figure you submit, use r=1, k=5, cycles=1
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  %%%% PART 2 code goes just below these comments - YOU NEED TANGENT VECTORS
  %%%% for this to work so right now this is commented out. Uncomment the
  %%%% code below when you're ready to work on this part.
    
  C=cross([xa ya za],[tx ty tz]);
  C=normalize(C);
  D=cross([tx ty tz], C);
  D=normalize(D);
  	vx=C(1);
  	vy=C(2);	% These are dummy vectors, replace with the
  	vz=C(3);	% correct ones!
  
  	ux=D(1);
  	uy=D(2);
  	uz=D(3);
  
  	v=[vx vy vz];
  	u=[ux uy uz];
  
  	% Let's plot a parametric circle using these vectors.
  	% it they are correct, you should get circles that
  	% are perpendicular to the curve at each of the points
  	% where we computed a tangent vector.
    % To do this, we're using a parametric circle
    %
    %   C(t)=v*cos(t) + u*sin(t), t in [0, 2*pi]
    %
    % and since the circle is centered at point p
    % on the curve, we have:
    %
    %   C(t)=p + v*cos(t) + u*sin(t), t in [0, 2*pi]
    %
    % It looks weird below because I'm using Matlab
    % operators to get all points in one go without 
    % using loops, but I'm just evaluating the
    % equation above! (yes, really!).cl
  
    	t_circ=[0:.1:2*pi]';
        rad=.5;				% Circle radius!
    	p_circ=repmat([xa ya za],[length(t_circ) 1]);
        p_circ=p_circ+(repmat(rad*cos(t_circ),[1 3]).*repmat(v,[length(t_circ),1]));
        p_circ=p_circ+(repmat(rad*sin(t_circ),[1 3]).*repmat(u,[length(t_circ) 1]));
        plot3(p_circ(:,1),p_circ(:,2),p_circ(:,3),'.','color',[0 .75 0]);
        
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %% SAVE THIS FIGURE, you need to submit it. And the code below will change the plot!
  %%
  %% For the figure you submit, use r=1, k=5, cycles=1
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
end;
