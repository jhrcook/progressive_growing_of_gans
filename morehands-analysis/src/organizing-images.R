
crop_subimage <- function(fname, x, y, rows, cols) {
  im <- imager::load.image(fname)[, , 1, 1]
  width <- dim(im)[1] / cols
  height <- dim(im)[2] / rows
  x_crop <- (x * width):((x + 1) * width)
  y_crop <- (y * height):((y + 1) * height)
  cropped_im <- imager::as.cimg(im[x_crop, y_crop])
  return(cropped_im)
}

make_training_video <- function(dir, crop_x = NA, crop_y = NA, rows = NA, cols = NA) {
  img_files <- list.files(dir, pattern = "png$", full.names = TRUE)
  img_files <- img_files[!stringr::str_detect(img_files, "reals")]
  tdir <- tempdir()
  for (i in seq(1, length(img_files))) {
    fname <- img_files[[i]]
    if (!is.na(crop_x) & !is.na(crop_y)) {
      im <- crop_subimage(fname, x = crop_x, y = crop_y, rows = rows, cols = cols)
    } else {
      im <- imager::load.image(fname)
    }
    padded_i <- stringr::str_pad(i, width = 5, side = "left", pad = "0")
    save_fname <- file.path(tdir, glue::glue("image-{i}.png"))
    imager::save.image(im, save_fname)
  }
  imager::make.video(tdir, "example-video.mpeg", verbose=TRUE)
}

make_training_video(res_dir, crop_x = 0, crop_y=0, rows=2, cols=3)

res_dir <- file.path("../results/004-pgan-hand-radiographs-preset-v2-4gpus-fp16-HIST")
crop_subimage(file.path(res_dir, "fakes010838.png"), x = 0, y = 0, rows = 2, cols = 3)
