## Demo file

# wkdir <- 'ICF_autocapsule_disabed'
# setwd( wkdir )
libdir <- 'd:/ICF_AutoCapsule_disabled'
library(R0, lib.loc = libdir)


wk2dir <- 'd:/ICF_AutoCapsule_disabled/covid/covmodel'
setwd( wk2dir )


covid_file <- 'd:/ICF_AutoCapsule_disabled/covid/covmodel/data/covid-19-4-sjis.csv'
covid_file <- 'd:/ICF_AutoCapsule_disabled/covid/covid19/data/covid-19-4-sjis.csv'

covid <- read.csv(covid_file)
fixed <- covid[,'Šm’è“ú']
mtable <- table(fixed)  # value count table
mdf    <- as.data.frame(mtable)
sorted <- mdf[order(as.Date(mdf$fixed, format='%m/%d/%Y')),]
sorted_len <- length(sorted$fixed)
# sorted2 <- sorted[1:sorted_len-1,]
# sconfirmed <- sorted2[sorted_len-30:sorted_len-1,2]
sorted_len

# sorted2 <- sorted[sorted_len-30:sorted_len-1,2]
sorted2 <- sorted[1:99,2]

epidata <- sorted2


# Generating an epidemic with given parameters
# mGT <- generation.time("gamma", c(3,1.5))
mGT <- generation.time("gamma", c(4,1.5))

mEpid <- sim.epid(epid.nb=1, GT=mGT, epid.length=30, family="poisson", R0=1.67, peak.value=50000)
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
