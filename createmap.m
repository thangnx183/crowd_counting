for i = 1:300
img = strcat(img_path,num2str(i),'.jpg')
head = strcat(gt_path,num2str(i),'.mat')
dmap = strcat(dmap_path,num2str(i),'.mat')
d = density(img,head,7)
sum(sum(d))
save dmap d
endfor