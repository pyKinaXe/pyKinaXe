suppressMessages({library(tiff)})
f <- "data/benchmarking_data_set/641102408_641102409-on 1200PTKlysv04-run 240822144936/ImageResults/641102408_W1_F1_T200_P94_I317_A30.tif"
img <- readTIFF(f)
sm  <- t(as.matrix(imager::isoblur(imager::as.cimg(t(img)), 2, gaussian=TRUE)))
cap <- quantile(sm, 0.98); smc <- sm; smc[smc>cap] <- cap      # clip the control spots
pf  <- function(v){v <- v-median(v); v[v<0] <- 0; v}
fit <- function(pp,len,n){b<-list(s=-Inf)
  for(p in seq(21,22,by=.05)){hi<-len-p*(n-1)-30; if(hi<=30) next
    for(o in seq(30,hi,by=.5)){s<-mean(pp[round(seq(o,by=p,length.out=n))])-
      mean(pp[round(seq(o+p/2,by=p,length.out=n-1))]); if(s>b$s) b<-list(s=s,p=p,o=o)}}
  b}
px <- pf(rowSums(smc)); py <- pf(colSums(smc))
fx <- fit(px, nrow(smc), 14); fy <- fit(py, ncol(smc), 14)

png("figures/ptk_grid_method.png", width=1500, height=560, res=110)
par(mfrow=c(1,2), mar=c(4,4,3.2,1))
show <- function(prof, f, lab) {
  plot(seq_along(prof), prof, type="l", col="grey35", lwd=1.2, xlab=paste(lab,"(px)"),
       ylab="projection (clipped)", main=sprintf("%s: pitch %.2f px, origin %.1f px", lab, f$p, f$o))
  nodes <- seq(f$o, by=f$p, length.out=14)
  inter <- seq(f$o+f$p/2, by=f$p, length.out=13)
  abline(v=inter, col="#d9534f", lty=3)
  abline(v=nodes, col="#0275d8", lwd=2)
  legend("topright", c("14 comb teeth (spots)","interstices"), col=c("#0275d8","#d9534f"),
         lwd=c(2,1), lty=c(1,3), bty="n", cex=.9)
}
show(px, fx, "x axis"); show(py, fy, "y axis")
invisible(dev.off()); cat("ok\n")
