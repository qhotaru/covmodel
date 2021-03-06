## Demo file

# wkdir <- 'ICF_autocapsule_disabed'
# setwd( wkdir )
libdir <- 'd:/ICF_AutoCapsule_disabled'
library(R0, lib.loc = libdir)
library(jsonlite, lib.loc=libdir)
#
#
#
wk2dir <- 'd:/ICF_AutoCapsule_disabled/covid/covmodel'
setwd( wk2dir )
#
#
# covid_file <- 'd:/ICF_AutoCapsule_disabled/covid/covmodel/data/covid-19-4-sjis.csv'
covid_file <- 'd:/ICF_AutoCapsule_disabled/covid/covid19/data/data.json'
covid_file <- 'd:/ICF_AutoCapsule_disabled/covid/covmodel/tokyo.csv'

# json  <- jsonlite::read_json( covid_file )
# pdata <- json$patients_summary$data
# pdf   <- data.frame(pdata)

# exit()
#
#
#
covid <- read.csv(covid_file)

fixed <- covid$positive
dates <- covid$date

epidata <- fixed

# Generating an epidemic with given parameters
# mGT <- generation.time("gamma", c(3,1.5))
mGT <- generation.time("gamma", c(4,1.5))

mEpid <- sim.epid(epid.nb=1, GT=mGT, epid.length=length(covid$positive), family="poisson", R0=1.67, peak.value=50000)
mEpid <- mEpid[,1]

# Running estimations
# est <- estimate.R(epid=mEpid, GT=mGT, methods=c("EG","ML","TD"))
est <- estimate.R(epid=epidata, GT=mGT, methods=c("EG","ML","TD"))

# Model estimates and goodness of fit can be plotted
plot(est)
plotfit(est)

# Sensitivity analysis for the EG estimation; influence of begin/end dates
s.a <- sensitivity.analysis(res=est$estimates$EG, begin=1:15, end=16:30, sa.type="time")

# This sensitivity analysis can be plotted
plot(s.a)
