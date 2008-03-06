clear;

S = 50;                     % # of samples around each boundary
DO_INVERSE = true;

warning off;
I=imread('PyAMG.png');
BW = im2bw(I);              %convert to Black & White
if DO_INVERSE
    BW = ~BW;               %invert image
end

subplot(2,2,1); imshow(I);

%BW = imfill(BW,'holes');   %remove holes

[B,L,N,A] = bwboundaries(BW,4);

subplot(2,2,2);
imshow(I)
hold on;
% show boundaries
for k=1:length(B)
    boundary = B{k};
    if(k > N)
        plot(boundary(:,2), boundary(:,1),'g','LineWidth',2);
    else
        plot(boundary(:,2), boundary(:,1),'r','LineWidth',2);
    end
end

XX = [];
YY = [];
EE = [];
for k=1:length(B)
    X = B{k}(:,2);
    Y = size(I,1) - B{k}(:,1);
    
    T  = linspace(0,2*pi,length(X));
    TT = linspace(0,2*pi,S);

    X = spline(T,X,TT);
    Y = spline(T,Y,TT);
    S = S - 1;
    
    X = X(1:S);
    Y = Y(1:S);
    
    XX = [XX X];
    YY = [YY Y];
  
    E = size(EE,2) + [ [S 1:(S-1)]; 1:S ];
    EE = [EE E];
end
subplot(2,2,3), 
triplot([EE; EE(1,:)]',XX,YY)

if DO_INVERSE
    holes = [ [75 160]; [235 140]]';
else
    holes = [ [60 130]; [150 125]; [210 130]; [290 150]; [400 140]]';
end


%%% doesn't work reliably
% holes = [];
% for k=N+1:length(B)
%     [i,j] = find(L == k,1)    
%     x = j;
%     y = size(I,1) - i;
%     holes = [holes [x,y]'];
% end

%%% Format of .poly file
% First line: <# of vertices> <dimension (must be 2)>  <# of attributes> <# of boundary markers (0 or 1)>
% Following lines: <vertex #> <x> <y> [attributes] [boundary marker]
% One line: <# of segments> <# of boundary markers (0 or 1)>
% Following lines: <segment #> <endpoint> <endpoint> [boundary marker]
% One line: <# of holes>
% Following lines: <hole #> <x> <y>
% Optional line: <# of regional attributes and/or area constraints>
% Optional following lines: <region #> <x> <y> <attribute> <maximum area> 
FID=fopen('Z.poly','w');
fprintf(FID,'%d %d %d %d\n',length(XX),2,0,0);
fprintf(FID,'%d %g %g\n',[1:length(XX);XX;YY]);
fprintf(FID,'%d %d\n',length(XX),0);
fprintf(FID,'%d %d %d\n',[1:size(EE,2); EE]);
fprintf(FID,'%d\n',size(holes,2));
fprintf(FID,'%d %g %g\n',[1:size(holes,2); holes]);
fclose(FID);

% now run "./triangle -q -p Z.poly" to run a conforming Delaunay triang
%
unix('./triangle -q -p -a2 Z.poly');
%unix('./showme Z.1.ele');

% now parse Z.1.node and Z.1.ele
FID=fopen('Z.1.node','r');
tmp=fscanf(FID,'%d',4);

fN = tmp(1);
fALL = fscanf(FID,'%d %g %g %g',[4 inf]);
fX = fALL(2,:)';
fY = fALL(3,:)';
fclose(FID);
FID=fopen('Z.1.ele','r');
tmp=fscanf(FID,'%d',3);
tri = fscanf(FID,'%d %d %d %d',[4 inf]);
tri=tri(2:4,:)';
subplot(2,2,4),trimesh(tri,fX,fY,'Color','k');


% write out results
elements = tri - 1;
vertices = [fX,fY];

save 'pyamg.mat' elements vertices;
