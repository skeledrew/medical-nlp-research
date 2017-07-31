# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# User specific aliases and functions
export CTAKES_HOME=/home/aphillips5/ctakes/trunk/
export JAVA_HOME=/opt/jdk1.8.0_121/
export DATA_DIR=/NLPShare/Alcohol/data/
export PATH="/home/aphillips5/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
csv () { cat $1 | tr -cd '\11\12\40-\176' | sed -e 's/,,/, ,/g' | column -s, -t | less -#5 -N -S; }  # mod to ignor 0 arg
source /home/aphillips5/envs/nlpenv/bin/activate
empty () { rm $1/yes/*; rm $1/no/*; }
UmlsLP () { mvn exec:java -Dexec.mainClass="org.apache.ctakes.pipelines.UmlsLookupPipeline" -Dexec.args="--input-dir $1/yes/ --output-dir $2/yes/" -Dctakes.umlsuser=<user> -Dctakes.umlspw=<pass> > out.tmp ; mvn exec:java -Dexec.mainClass="org.apache.ctakes.pipelines.UmlsLookupPipeline" -Dexec.args="--input-dir $1/no/ --output-dir $2/no/" -Dctakes.umlsuser=<user> -Dctakes.umlspw=<pass> >> out.tmp ; echo Finished UmlsLP $1 $2 ; cat out.tmp | grep -C 3 "Total time" ; rm out.tmp ;}
ExtractCuis () { mvn exec:java -Dexec.mainClass="org.apache.ctakes.consumers.ExtractCuis" -Dexec.args="--xmi-dir $1/no/ --output-dir $2/no/" > tmp.out ; mvn exec:java -Dexec.mainClass="org.apache.ctakes.consumers.ExtractCuis" -Dexec.args="--xmi-dir $1/yes/ --output-dir $2/yes/" >> tmp.out ; echo Finished ExtractCuis $1 $2 ; cat tmp.out |grep -C 3 "Total time" ; rm tmp.out ;}
