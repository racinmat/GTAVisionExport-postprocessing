in_directory = './../output';
name = 'info-2017-11-14--12-03-12';
name = 'info-2017-11-14--12-06-33';
t = Tiff(fullfile(in_directory, [name '.tiff']),'r');
im = t.read();
im = im(:,:,1:3);
t_depth = Tiff(fullfile(in_directory, [name '-depth.tiff']),'r');
im_depth = t_depth.read();
t_stencil = Tiff(fullfile(in_directory, [name '-stencil.tiff']),'r');
im_stencil = t_stencil.read();
im_stencil_ids = mod(im_stencil, 16); % because lower 4 bits are used for object IDs
im_stencil_flags = im_stencil - im_stencil_ids;
figure()
imagesc(im);
figure()
colormap(gray)
imagesc(im_depth);
figure()
imagesc(im_stencil);
figure()
imagesc(im_stencil_ids);
figure()
imagesc(im_stencil_flags);
% imshow(im);
% imshow(im_depth);
% imshow(im_stencil);
