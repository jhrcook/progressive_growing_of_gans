# Extract evaluation metric results into data frames.

source("src/extract-logged-data.R")

parse_matrix_log <- function(f, start_pattern, n_splits) {
  f_lines <- unlist(readLines(f))
  start <- which(stringr::str_detect(f_lines, start_pattern))
  f_lines <- f_lines[start:length(f_lines)]
  f_lines <- f_lines[-2]
  finish <- which(f_lines == "") - 1
  f_lines <- f_lines[1:finish]
  f_lines <- stringr::str_split_fixed(f_lines, "[:space:]{2,}", n = n_splits)
  cols <- unlist(f_lines[1, ]) %>%
    paste(collapse = " ") %>%
    stringr::str_split("[:space:]+") %>%
    unlist()
  f_lines <- f_lines[-1, ]
  df <- tibble::tibble(as.data.frame(f_lines))
  colnames(df) <- cols
  return(df)
}

extract_fid <- function(f) {
  df <- parse_matrix_log(f, "Snapshot", 3)
  df <- df %>%
    janitor::clean_names() %>%
    dplyr::slice(-1) %>%
    dplyr::mutate(time_eval = unlist(purrr::map(time_eval, convert_log_time)))
  return(df)
}

extract_is <- function(f) {
  warning(
    "This function has not been tested on real files, so it may not work properly"
  )
  df <- parse_matrix_log(f, "Snapshot", 4)
  df <- df %>%
    janitor::clean_names() %>%
    dplyr::slice(-1) %>%
    dplyr::mutate(time_eval = unlist(purrr::map(time_eval, convert_log_time)))
}

extract_msssim <- function(f) {
  df <- parse_matrix_log(f, "Snapshot", 3)
  df <- df %>%
    janitor::clean_names() %>%
    dplyr::slice(-1) %>%
    dplyr::mutate(time_eval = unlist(purrr::map(time_eval, convert_log_time)))
  return(df)
}

extract_swd <- function(f) {
  df <- parse_matrix_log(f, "Snapshot", 10)
  df <- df %>%
    janitor::clean_names() %>%
    dplyr::slice(-1) %>%
    dplyr::mutate(time_eval = unlist(purrr::map(time_eval, convert_log_time)))
  return(df)
}

## EXAMPLES
# res_dir <- file.path("../results/004-pgan-hand-radiographs-preset-v2-4gpus-fp16-HIST")
# fid_results <- extract_fid(file.path(res_dir, "metric-fid-10k.txt"))
# is_results <- extract_is(file.path(res_dir, "metric-is-50k.txt"))
# msssim_results <- extract_msssim(file.path(res_dir, "metric-msssim-20k.txt"))
# swd_results <- extract_swd(file.path(res_dir, "metric-swd-16k.txt"))
