libdir <- 'd:/ICF_AutoCapsule_disabled'
wkdir <- libdir
library(jsonlite, lib.loc = libdir)

tokyo <- 'd:/ICF_AutoCapsule_disabled/covid19/data/data.json'

json1 <- read_json(tokyo)
str(json1)

# simplify
simple1 <- read_json(tokyo, simplifyVector = TRUE)
str(simple1)

# write

tmp <- tmpfile()
data1 <- c(1,2,3,4,5)
write_json(data1, tmp)

tmp3 <- tmpfile()
write_json(iris, tmp3)

readr::read_file(tmp)

tmp2 <- tmpfile()
write_json(data1, tmp2, pretty = FALSE)
cat(readr::read_lines(tmp2,n_max=10L), sep='\n')

c <- read_json(tmp2)
str(c)

#
#
#
d <- read_json(tmp2, simplifyvector = TRUE)
str(d)

