# Extract the training info from a log file.

library(magrittr)

log_col_names <- c(
  "tick",
  "kimg",
  "lod",
  "minibatch",
  "time",
  "sec_per_tick",
  "sec_per_kimg",
  "maintenance"
)

convert_log_time <- function(t) {
  to_num <- function(x) {
    x <- as.numeric(x)
    if (is.na(x)) {
      return(0)
    } else {
      return(x)
    }
  }

  days <- to_num(stringr::str_extract(t, {"[:digit:]+(?=d)"})) * 24 * 60 * 60
  hours <- to_num(stringr::str_extract(t, {"[:digit:]+(?=h)"})) * 60 * 60
  minutes <- to_num(stringr::str_extract(t, {"[:digit:]+(?=m)"})) * 60
  seconds <- to_num(stringr::str_extract(t, {"[:digit:]+(?=s)"}))
  return(
    lubridate::duration(days + hours + minutes + seconds, units = "seconds")
  )
}

parse_log_time <- function(r) {
  t <- stringr::str_extract(r, "(?<=time ).+(?=sec/tick)")
  return(convert_log_time(t))
}

parse_log_row <- function(r) {
  tick <- stringr::str_extract(r, "(?<=tick )[:digit:]+")
  kimg <- stringr::str_extract(r, "(?<=kimg )[:digit:]+")
  lod <- stringr::str_extract(r, "(?<=lod )[:digit:]+")
  minibatch <- stringr::str_extract(r, "(?<=minibatch )[:digit:]+")
  time <- parse_log_time(r)
  sec_per_tick <- stringr::str_extract(r, "(?<=sec/tick )[:digit:]+")
  sec_per_kimg <- stringr::str_extract(r, "(?<=sec/kimg )[:digit:]+")
  maintenance <- stringr::str_extract(r, "(?<=maintenance )[:digit:]+")
  return(tibble::tibble(
    tick=as.numeric(tick),
    kimg=as.numeric(kimg),
    lod = as.numeric(lod),
    minibatch = as.numeric(minibatch),
    time = as.numeric(time),
    sec_per_tick = as.numeric(sec_per_tick),
    sec_per_kimg = as.numeric(sec_per_kimg),
    maintenance = as.numeric(maintenance)
  ))
}

extract_logged_data <- function(f) {
  f_lines <- unlist(readLines(f))
  start <- which(stringr::str_detect(f_lines, "Training...")) + 1
  f_lines <- f_lines[start:length(f_lines)]
  purrr::map_dfr(f_lines, parse_log_row)
}

# a <- extract_logged_data(
#   "../results/011-pgan-hand-radiographs-preset-v2-4gpus-fp16-HIST/log.txt"
# )
