% the data that you want to plot as a 3D surface.
[x,y] = meshgrid(-10:0.1:10);
z = (abs(x.^3) + abs(y.^3)).^0.5 +10;
 
% get the corners of the domain in which the data occurs.
min_x = min(min(x));
min_y = min(min(y));
max_x = max(max(x));
max_y = max(max(y));
 
% the image data you want to show as a plane.
planeimg = abs(z);
 
% set hold on so we can show multiple plots / surfs in the figure.
figure; hold on;
 
% do a normal surface plot.
surf(x,y,z,'FaceColor','interp',...
   'EdgeColor','none',...
   'FaceLighting','phong')

axis off;
%mesh(x,y,z)
 
% plot the image plane using surf.
surf([min_x max_x],[min_y max_y],repmat(imgzposition, [2 2]),...
    planeimg,'facecolor','texture','EdgeColor','none')
 
% set a colormap for the figure.
colormap(jet);
 
% set the view angle.
view(45,30);
 