from std.algorithm import parallelize

def main():
    var n = 10
    var k = 5
    
    @parameter
    def process(i: Int):
        print(i, k)
        
    parallelize[process](n, n)
