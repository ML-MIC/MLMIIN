set.seed(2024)
n = 1200
p <- mlbench::mlbench.spirals(n, cycles=2, sd=0.06)
plot(p)
write.csv(data.frame(X0 = p$x[,1], X1 = p$x[,2], Y =( p$classes == 1)*1), 
          file = "./spirals.csv", row.names = FALSE)

