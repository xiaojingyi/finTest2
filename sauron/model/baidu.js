var dic = Array();

var persons = $(".all-star-wall .img-item span");
for(var i = 0; i < persons.length; ++i){ 
    dic[$(persons[i]).text()] = $(persons[i]).text();
    $(persons[i]).css("font-weight", "bold");
}
var persons = $(".sub-view span");
for(var i = 0; i < persons.length; ++i){ 
    dic[$(persons[i]).text()] = $(persons[i]).text();
    $(persons[i]).css("font-weight", "bold");
}
var persons = $(".text-item");
for(var i = 0; i < persons.length; ++i){ 
    dic[$(persons[i]).text()] = $(persons[i]).text();
    $(persons[i]).css("font-weight", "bold");
}

for (var k in dic){
    console.log("'"+k+"',");
}
