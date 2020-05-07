listings = dir('*.ncl')
criteriaSum = 0;
BB = [];
II = [];
JJ = [];

for i=1:length(listings)
    disp(listings(i).name)
    filename = listings(i).name
    
    ncdisp(filename);
    B = ncread(filename,'B');
    I = ncread(filename,'I');
    J = ncread(filename,'J');
    BB = vertcat(BB,B);
    II = vertcat(II,I);
    JJ = vertcat(JJ,J);
    criteriaSum = criteriaSum + length(B);
end

scatter(II,JJ)
fprintf("number of points that met the criteria=%i\n",criteriaSum)
fprintf("fraction of points meeting the criteria=%f\n",double(criteriaSum)/51196^2)

