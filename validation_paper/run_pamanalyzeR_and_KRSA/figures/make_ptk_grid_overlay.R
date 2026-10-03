suppressMessages({library(pamgeneAnalyzeR); library(tiff)})
load_tiff_as_matrix <- function(p) structure(unclass(tiff::readTIFF(p)), class="matrix")
locate <- function(img, n, pr=c(21,22), margin=30, cq=0.98) {
  sm <- t(as.matrix(imager::isoblur(imager::as.cimg(t(img)), 2, gaussian=TRUE)))
  cap <- quantile(sm, cq, na.rm=TRUE); sm[sm>cap] <- cap
  pf <- function(v){v <- v-median(v); v[v<0]<-0; v}
  fit <- function(pp,len){b<-list(s=-Inf)
    for(p in seq(pr[1],pr[2],by=.05)){hi<-len-p*(n-1)-margin; if(hi<=margin) next
      for(o in seq(margin,hi,by=.5)){s<-mean(pp[round(seq(o,by=p,length.out=n))])-
        mean(pp[round(seq(o+p/2,by=p,length.out=n-1))]); if(s>b$s) b<-list(s=s,p=p,o=o)}}
    b}
  fx<-fit(pf(rowSums(sm)),nrow(sm)); fy<-fit(pf(colSums(sm)),ncol(sm))
  list(px=fx$p,x0=fx$o,py=fy$p,y0=fy$o)
}
d <- "data/benchmarking_data_set/641102408_641102409-on 1200PTKlysv04-run 240822144936/ImageResults"
files <- sort(list.files(d))
img_first <- load_tiff_as_matrix(file.path(d, files[1]))     # their reference: files[1]
img_bright <- load_tiff_as_matrix(file.path(d, "641102408_W1_F1_T200_P94_I317_A30.tif"))
cat("their reference image (files[1]):", files[1], "\n")

cc <- find_centers(img_first)
pep <- cc[grepl("^Peptide", cc$names), ]
g <- locate(img_bright, 14)
xs <- seq(g$x0, by=g$px, length.out=14); ys <- seq(g$y0, by=g$py, length.out=14)
grid14 <- expand.grid(x=xs, y=ys)

png("figures/ptk_grid_overlay.png", width=1500, height=620, res=110)
par(mfrow=c(1,2), mar=c(1,1,3,1))
sh <- function(m) { m <- (m-min(m))/(max(m)-min(m)); m <- m^0.5
  image(t(m[nrow(m):1,]), col=grey.colors(256), axes=FALSE, useRaster=TRUE) }
toU <- function(v, n) (v-1)/(n-1)
sh(img_bright); title("find_centers(): 144 points (12x12)")
points(toU(pep$y, ncol(img_bright)), 1-toU(pep$x, nrow(img_bright)),
       col="red", pch=3, cex=.7, lwd=2)
sh(img_bright); title("detected 14x14 grid: 196 points")
points(toU(grid14$y, ncol(img_bright)), 1-toU(grid14$x, nrow(img_bright)),
       col="cyan", pch=3, cex=.7, lwd=2)
invisible(dev.off())
cat("wrote figures/ptk_grid_overlay.png\n")
