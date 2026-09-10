// Node-side check for the PDF viewer's search matcher.
//
// The viewer itself needs a DOM and a real PDF, so it is verified in the browser
// (see the project's browser probes). What a screenshot cannot show is whether a
// chunk's text still matches the page text *after* extraction has broken words
// across lines — so that rule is exercised directly here.
//
// `pdfview.js` imports pdf.js from its served URL (browser-absolute, and it needs
// a DOM), so this reads the file, evaluates the matcher's own source, and runs the
// cases below against it: the code under test is the code in the file.
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const SOURCE = join(HERE, "..", "..", "raggy", "gui", "web", "js", "pdfview.js");

function extractFunction(name) {
  const text = readFileSync(SOURCE, "utf8");
  const start = text.indexOf(`function ${name}(`);
  if (start === -1) throw new Error(`${name} not found in ${SOURCE}`);
  let depth = 0;
  let started = false;
  for (let index = start; index < text.length; index += 1) {
    const character = text[index];
    if (character === "{") {
      depth += 1;
      started = true;
    } else if (character === "}") {
      depth -= 1;
      if (started && depth === 0) return text.slice(start, index + 1);
    }
  }
  throw new Error(`could not find the end of ${name}`);
}

const matchFlexible = new Function(
  `${extractFunction("matchFlexible")}; return matchFlexible;`,
)();

// `hay` is what the text layer concatenates to (pdf.js emits one span per text
// run and the viewer joins them with a space); `needle` is a chunk's own text as
// raggy stores it. `want` counts returned ranges: one per needle word matched,
// which the highlight merges into one mark per text span.
const cases = [
  ["plain phrase", "TS-RAG is a retrieval augmented generation method", "retrieval augmented generation", 3],
  ["an acronym's hyphen is not a word break", "TS-RAG is a retrieval augmented", "retrieval augmented", 2],
  ["line break inside the phrase", "we propose TS-RAG, a novel\nframework that leverages", "a novel framework that leverages", 5],
  ["hyphenated word, space after the hyphen", "we com- pute the score from the data", "we compute the score", 4],
  ["hyphenated word, no space", "improve effi-ciency, Autoformer leverages", "improve efficiency, Autoformer", 3],
  ["hyphen and newline together", "comparison of ex-\nisting methods", "comparison of existing methods", 4],
  ["every occurrence is found", "retrieval augmented generation and retrieval augmented generation", "retrieval augmented generation", 6],
  ["case insensitive", "TS-RAG: Retrieval Augmented", "ts-rag: retrieval augmented", 3],
  ["absent text does not match", "the quick brown fox", "lazy dog", 0],
  ["reordered words do not match", "generation augmented retrieval", "retrieval augmented generation", 0],
  ["a dropped internal word does not match", "we compute the score for the whole corpus", "we compute score", 0],
  ["no matching inside a longer word", "the computer computes a compute", "compute", 1],
  ["every standalone occurrence", "compute it, computer, compute", "compute", 2],
  ["a slash is not a word break", "either/or choices ahead", "either or choices", 0],
  ["a comma is not a word break", "first, second, third", "first second", 0],
  ["punctuation the needle also has", "TS-RAG: Retrieval Augmented Generation", "ts-rag: retrieval augmented generation", 4],
  ["word at the very start", "retrieval works", "retrieval", 1],
  ["word at the very end", "we do retrieval", "retrieval", 1],
  ["one-word needle inside a phrase", "we compute the score", "compute", 1],
  ["hyphen inside the searched word", "we com-pute the score", "we compute the score", 4],
];

let failures = 0;
for (const [name, hay, needle, want] of cases) {
  const got = matchFlexible(hay, needle).length;
  const ok = got === want;
  if (!ok) failures += 1;
  console.log(`${ok ? "ok  " : "FAIL"} ${name}: got ${got}, want ${want}`);
}

// The ranges have to point at the matched text, not merely count it.
for (const [hay, needle, expected] of [
  ["prefix we com- pute suffix", "we compute", /we com-\s*pute/],
  ["before ex-\nisting after", "existing", /ex-\nisting/],
  ["x retrieval augmented generation y", "retrieval augmented generation", /^retrieval augmented generation$/],
]) {
  const ranges = matchFlexible(hay, needle);
  const matched = ranges.length ? hay.slice(ranges[0][0], ranges[ranges.length - 1][1]) : "";
  const ok = expected.test(matched);
  if (!ok) failures += 1;
  console.log(`${ok ? "ok  " : "FAIL"} range covers the matched text: ${JSON.stringify(matched)}`);
}

console.log(`\nfailures: ${failures}`);
process.exit(failures === 0 ? 0 : 1);
