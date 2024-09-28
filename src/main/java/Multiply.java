import java.io.*;
import java.util.*;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.util.*;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.*;
import org.apache.hadoop.mapreduce.lib.output.*;

class Triple implements Writable {
    public short tag; // 0 for Matrix M, 1 for Matrix N
    public int index; // index k the middle index column in M and row in N
    public float value; // actual value at position (i,k) or (k,j)

    public Triple() {
        tag = 0;
        index = 0;
        value = 0.0f;
    }


    public Triple(short tag, int index, float value) {
        this.tag = tag;
        this.index = index;
        this.value = value;
    }

    @Override
    public void write(DataOutput out) throws IOException {
        out.writeShort(tag);
        out.writeInt(index);
        out.writeFloat(value);
    }


    @Override
    public void readFields(DataInput in) throws IOException {
        tag = in.readShort();
        index = in.readInt();
        value = in.readFloat();
    }


}

class Pair implements WritableComparable<Pair> {
    public int i;
    public int j;


    public Pair() {
        i = 0;
        j = 0;
    }


    public Pair(int i, int j) {
        this.i = i;
        this.j = j;
    }


    @Override
    public void write(DataOutput out) throws IOException {
        out.writeInt(i);
        out.writeInt(j);
    }


    @Override
    public void readFields(DataInput in) throws IOException {
        i = in.readInt();
        j = in.readInt();
    }

    @Override
    public String toString() {
        return i + " " + j + " ";
    }

    @Override
    public int compareTo(Pair o) {
        int result = Integer.compare(this.i, o.i);
        if (result != 0) {
            return result;
        }
        return Integer.compare(this.j, o.j);
    }

    @Override
    public int hashCode() {
        return Objects.hash(i, j);
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        Pair other = (Pair) obj;
        return this.i == other.i && this.j == other.j;
    }
}

public class Multiply {

    public static class MapperMatrixM extends Mapper<Object, Text, IntWritable, Triple> {

        @Override
        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {

            String line = value.toString();
            String[] stringTokens = line.split(",");
            int i = Integer.parseInt(stringTokens[0]); // index i in Matrix M (i,k)
            float cellValue = Float.parseFloat(stringTokens[2]);
            IntWritable keyIndex = new IntWritable(Integer.parseInt(stringTokens[1])); // index k in Matrix M (i,k)
            Triple t = new Triple((short) 0, i, cellValue);
            context.write(keyIndex, t);

        }
    }

    public static class MapperMatrixN extends Mapper<Object, Text, IntWritable, Triple> {

        @Override
        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {

            String line = value.toString();
            String[] stringTokens = line.split(",");
            int j = Integer.parseInt(stringTokens[1]); // index j in matrix N (k, j)
            float cellValue = Float.parseFloat(stringTokens[2]);
            IntWritable keyIndex = new IntWritable(Integer.parseInt(stringTokens[0])); //index k in matrix N (k, j)
            Triple t = new Triple((short) 1, j, cellValue);
            context.write(keyIndex, t);

        }
    }

    public static class ReducerCombine extends Reducer<IntWritable, Triple, Pair, DoubleWritable> {

        @Override
        public void reduce(IntWritable key, Iterable<Triple> values, Context context)
                throws IOException, InterruptedException {

            ArrayList<Triple> M = new ArrayList<Triple>();
            ArrayList<Triple> N = new ArrayList<Triple>();


            for (Triple entry : values) {

                Triple tempTriple = ReflectionUtils.newInstance(Triple.class, context.getConfiguration());
                ReflectionUtils.copy(context.getConfiguration(), entry, tempTriple);

                if (tempTriple.tag == 0) {
                    M.add(tempTriple);
                } else if (tempTriple.tag == 1) {
                    N.add(tempTriple);
                }
            }

            for (Triple mElement : M) {
                for (Triple nElement : N) {
                    // (i, j) where i comes from M and j comes from N
                    Pair p = new Pair(mElement.index, nElement.index);
                    double product = mElement.value * nElement.value;
                    context.write(p, new DoubleWritable(product));
                }
            }
        }
    }

    public static class AdditionMapper extends Mapper<Object, Text, Pair, DoubleWritable> {
        @Override
        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {

            String[] pairValue = value.toString().split(" ");
            int first = Integer.parseInt(pairValue[0]);
            int second = Integer.parseInt(pairValue[1]);
            double val = Double.parseDouble(pairValue[2]);
            Pair p = new Pair(first, second);
            context.write(p, new DoubleWritable(val));
        }
    }

    public static class AdditionReducer extends Reducer<Pair, DoubleWritable, Text, DoubleWritable> {
        @Override
        public void reduce(Pair key, Iterable<DoubleWritable> values, Context context)
                throws IOException, InterruptedException {

            double total = 0.0;
            for (DoubleWritable value : values) {
                total += value.get();
            }
            String formattedKey = String.format("(%d,%d)", key.i, key.j);
            context.write(new Text(formattedKey), new DoubleWritable(total));
        }
    }

    public static void main(String[] args) throws Exception {
        // Job 1 First MapReduce Job
        Job job1 = Job.getInstance();
        job1.setJobName("IntermediateJob");
        job1.setJarByClass(Multiply.class);

        MultipleInputs.addInputPath(job1, new Path(args[0]), TextInputFormat.class, MapperMatrixM.class); // Matrix M
        MultipleInputs.addInputPath(job1, new Path(args[1]), TextInputFormat.class, MapperMatrixN.class); // Matrix N

        job1.setReducerClass(ReducerCombine.class);
        job1.setMapOutputKeyClass(IntWritable.class);
        job1.setMapOutputValueClass(Triple.class);
        job1.setOutputKeyClass(Pair.class);
        job1.setOutputValueClass(DoubleWritable.class);
        job1.setOutputFormatClass(TextOutputFormat.class);

        FileOutputFormat.setOutputPath(job1, new Path(args[2]));

        if (!job1.waitForCompletion(true)) {
            System.exit(1); // Exit if job1 fails
        }

        // Job 2 Second MapReduce addition of the multiplication products
        Job job2 = Job.getInstance();
        job2.setJobName("FinalMatrix");
        job2.setJarByClass(Multiply.class);

        job2.setMapperClass(AdditionMapper.class);
        job2.setReducerClass(AdditionReducer.class);

        job2.setMapOutputKeyClass(Pair.class);
        job2.setMapOutputValueClass(DoubleWritable.class);
        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(DoubleWritable.class);

        job2.setInputFormatClass(TextInputFormat.class);
        job2.setOutputFormatClass(TextOutputFormat.class);

        FileInputFormat.setInputPaths(job2, new Path(args[2]));
        FileOutputFormat.setOutputPath(job2, new Path(args[3]));

        if (!job2.waitForCompletion(true)) {
            System.exit(1); // Exit if job2 fails
        }
    }
}
