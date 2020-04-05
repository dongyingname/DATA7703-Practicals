function fnQ6 = out(in,n)
    
    size = length(in);
    fnQ6=zeros(1,size);
    %disp(fnQ6);
    bin = floor(size / n); %number of for loops used to inverse the vector
    residue = size - bin * n; %number of leftover numbers
    %disp(residue);
    
    for i = 1:bin
        %The numbers that are to be picked up from in and inserted into out
        inpick = in((size - i*n+1):(size - (i-1) * n));
        %disp(inpick)
        fnQ6(((i-1)*n+1):((i-1)*n+n)) = inpick;
        if (residue > 0)
            fnQ6((size-residue+1):end) = in(1:residue);
        end
    end
   
end