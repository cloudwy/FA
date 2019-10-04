x=-10:0.5:10;
y=-10:0.5:10;
mu=[0,0];
sigma=[5 0; 0 5]; 
[X,Y]=meshgrid(x,y); 
p=mvnpdf([X(:),Y(:)],mu,sigma);
P=reshape(p,size(X)); 
%figure(2)
surf(X,Y,P)
shading interp
%colorbar
%title('Draw samples from p(z)');