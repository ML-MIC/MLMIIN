library(fpp3)
library(dplyr)
library(purrr)
library(fma)

ts2csv <- function(objname, pname) {
  dir.create(paste0('data/', pname))
  fname <- paste0('data/', pname, '/', objname, '.csv')
  x = get(objname)
  tryCatch({
    readr::write_csv(as.data.frame(x), fname)
    print(fname)
  }, warning = function(w) {
  }, error = function(e) {
    print(paste0(fname, " <-- error"))
  }, finally = {
  })
}

for (pname in list("fpp3", "tsibbledata", "tsibble", "fma")) {
  # pname <- "fma"
  # objname <- x$results.Item[1]
  x <- as.data.frame(data(package=pname)[3])
  for (objname in x$results.Item) {
    ts2csv(objname, pname)
  }
}

data()

readr::write_csv(as_tibble(fma::sheep), 'data/fma/sheep.csv')

readr::write_csv(as_tibble(fma::bicoal), 'data/fma/bicoal.csv')