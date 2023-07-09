import { useEffect } from "react";
import { Voy } from "voy-search";
import { pipeline } from "@xenova/transformers";

export default function VoyExample() {
  const phrases = [
    "That is a very happy Person",
    "That is a Happy Dog",
    "Today is a sunny day",
  ];

  const query = "That is a happy person";

  const run = async () => {
    const embeddings = await Promise.all(phrases.map(calculateEmbeddings));

    const data = embeddings.map((embeddings, i) => ({
      id: String(i),
      title: phrases[i],
      url: `/path/${i}`,
      embeddings: Array.from(embeddings) as number[],
    }));

    console.log({ data });

    const index = new Voy({ embeddings: data });

    // Perform similarity search for a query embeddings
    const q = await calculateEmbeddings(query);
    console.log({ q });
    const result = index.search(q, 1);

    // Display search result
    result.neighbors.forEach((result) =>
      console.log(`✨ voy similarity search result: "${result.title}"`)
    );
  };

  useEffect(() => {
    run();
  }, []);

  return <h1>Embed</h1>;
}

async function calculateEmbeddings(text: string) {
  const extractor = await getPipeline();
  const result = await extractor(text, { pooling: "mean", normalize: true });
  return result.data;
}

async function getPipeline() {
  return await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
}
