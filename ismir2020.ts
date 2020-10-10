import * as fs from 'fs';
import * as _ from 'lodash';
import { QUANT_FUNCS as QF, getSimpleSmithWatermanPath, SmithWatermanOptions } from 'siafun';
import * as ms from 'musical-structure';


const DATASET = '../fifteen-songs-dataset/';
const AUDIO = DATASET+'tuned_audio/';
const FEATURES = DATASET+'features/';
const PATTERNS = DATASET+'patterns/';
const RESULTS = './results/';


const MSA_CONFIG: ms.MSAOptions　= {
  modelLength: ms.MSA_LENGTH.MEDIAN,
  iterations: 1,
  edgeInertia: 0.8,
  distInertia: 0.8,
  matchMatch: 0.999,
  deleteInsert: 0.01,
  flankProb: undefined
}

const FEATURE_CONFIG: ms.FeatureOptions = {
  selectedFeatures: [ms.FEATURES.MADMOM_BEATS, ms.FEATURES.GO_CHORDS]
}

const SW_CONFIG: SmithWatermanOptions = {
  quantizerFunctions: [QF.ORDER(), QF.IDENTITY()],
  maxIterations: 1,
  minSegmentLength: 16,
  nLongest: 10,
  maxGapSize: 4,
  maxGapRatio: 0.25,
  minDistance: 4,
  cacheDir: PATTERNS
}

const MAX_VERSIONS = 100;
const SELF_ALIGNMENTS = true;
const PAIRWISE_ALIGNMENTS = 0//5;
const NUM_CONNECTIONS = 1;
const MASK_THRESHOLD = .1;

console.log('hello')
run();
console.log('bye')

async function run() {
  const songs: string[] = _.keys(JSON.parse(
      fs.readFileSync(DATASET+'dataset.json', 'utf8')));
  const results = await ms.mapSeries(songs, async s => {
    console.log('working on '+s);
    const versions = ms.recGetFilesInFolder(AUDIO+s+'/', ['wav']).slice(-MAX_VERSIONS);
    const points = await new ms.FeatureLoader(FEATURES)
      .getPointsForAudioFiles(versions, FEATURE_CONFIG);
    console.log('saving feature sequences')
    const seqsFile = RESULTS+s+"-seqs.json";
    ms.saveMultinomialSequences(points, seqsFile);
    console.log('multiple sequence alignment')
    const msaFile = await ms.hmmAlign(seqsFile, RESULTS+s+'-', MSA_CONFIG);
    console.log('pairwise and self alignments')
    console.log({
      collectionName: s,
      patternsFolder: PATTERNS,
      algorithm: ms.AlignmentAlgorithm.SW,
      points: points,
      audioFiles: versions,
      swOptions: SW_CONFIG,
      includeSelfAlignments: SELF_ALIGNMENTS,
      numTuplesPerFile: PAIRWISE_ALIGNMENTS,
      tupleSize: 2
    })
    const alignments = ms.extractAlignments({
      collectionName: s,
      patternsFolder: PATTERNS,
      algorithm: ms.AlignmentAlgorithm.SW,
      points: points,
      audioFiles: versions,
      swOptions: SW_CONFIG,
      includeSelfAlignments: SELF_ALIGNMENTS,
      numTuplesPerFile: PAIRWISE_ALIGNMENTS,
      tupleSize: 2
    });
    return getEvaluation(s, points, msaFile, alignments);
  });
  
  ms.saveJsonFile(RESULTS+'eval.json', results);
}

async function getEvaluation(song: string, points: any[][][],
    msaFile: string, alignments: ms.Alignments) {
  //get original chord sequence
  const originalChords = points.map(ps => ps.map(p => ms.pcSetToLabel(p.slice(1)[0])));
  //calculate labels of harmonic essence
  const msaModeLabels = await ms.getTimelineModeLabels(points, msaFile, alignments);
  console.log(msaModeLabels)
  const graphBasedLabels = await ms.getTimelineSectionModeLabels(points, msaFile,
    alignments, NUM_CONNECTIONS, MASK_THRESHOLD);
  const timeline = (await ms.getPartitionFromMSAResult(points, msaFile, alignments))
    .getPartitions();
  //annotate individual versions
  const modeLabelChords = annotateIndividuals(originalChords, timeline, msaModeLabels);
  const graphLabelChords = annotateIndividuals(originalChords, timeline, graphBasedLabels);
  //align with leadsheets
  const leadsheet = DATASET+'leadsheets/'+song+'.json';
  const original = originalChords.map(c => getAlignmentPs(c, leadsheet));
  const tlModes = modeLabelChords.map(c => getAlignmentPs(c, leadsheet));
  const tlGraph = graphLabelChords.map(c => getAlignmentPs(c, leadsheet));
  const msa = getAlignmentPs(msaModeLabels, leadsheet);
  const graph = getAlignmentPs(graphBasedLabels, leadsheet);
  return {
    originalGround: _.mean(original.map(o => o.groundP)),
    originalSeq: _.mean(original.map(o => o.seqP)),
    tlModesGround: _.mean(tlModes.map(o => o.groundP)),
    tlModesSeq: _.mean(tlModes.map(o => o.seqP)),
    tlGraphGround: _.mean(tlGraph.map(o => o.groundP)),
    tlGraphSeq: _.mean(tlGraph.map(o => o.seqP)),
    msaGround: msa.groundP,
    msaSeq: msa.seqP,
    graphGround: graph.groundP,
    graphSeq: graph.seqP
  }
}

function annotateIndividuals(originalChords: string[][],
    timeline: ms.SegmentNode[][], labels: string[]) {
  return originalChords.map((cs,i) => cs.map((c,j) => {
    const index = timeline.findIndex(t =>
      t.find(n => n.version == i && n.time == j) != null);
    return index >= 0 ? labels[index] : c;
  }));
}

function getAlignmentPs(sequence: string[], leadSheetFile: string) {
  const groundtruth = ms.getStandardChordSequence(leadSheetFile, true);
  const vocab = _.uniq(_.concat(groundtruth, sequence));
  const numeric = (s: string[]) => s.map(v => [vocab.indexOf(v)]);
  const path = getSimpleSmithWatermanPath(numeric(groundtruth), numeric(sequence), {});
  return {groundP: path.length/groundtruth.length, seqP: path.length/sequence.length};
}

