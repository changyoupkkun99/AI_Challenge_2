import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {Workbook} from '@oai/artifact-tool';

const here=path.dirname(fileURLToPath(import.meta.url));
const plan=JSON.parse(await fs.readFile(path.join(here,'delivery_plan.json'),'utf8'));
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const raw=await fs.readFile(plan.baseline.path);
if(sha(raw)!==plan.baseline.sha256)throw Error('Baseline changed');
const wb=await Workbook.fromCSV(raw.toString('utf8'),{sheetName:'Submission'});
const sheet=wb.worksheets.getItemAt(0);
const original=sheet.getRange('A1:B6715').values;
if(JSON.stringify(original[0])!==JSON.stringify(['id','answer']))throw Error('Bad header');
const ids=original.slice(1).map(r=>r[0]);
if(new Set(ids).size!==6714)throw Error('Bad IDs');
const models=[];
for(const rec of plan.consensus_models){
  const b=await fs.readFile(rec.path);if(sha(b)!==rec.sha256)throw Error('Canonical base changed');
  const model=await Workbook.fromCSV(b.toString('utf8'),{sheetName:'Input'});
  const rows=model.worksheets.getItemAt(0).getRange('A1:B6715').values;
  if(rows.slice(1).some((r,i)=>r[0]!==ids[i]))throw Error('Different ID order');
  models.push(rows.slice(1).map(r=>r[1]));
}
const locked=ids.map((_,i)=>models.every(m=>m[i]===models[0][i]));
if(locked.filter(Boolean).length!==4444)throw Error('Wrong consensus mask');
const values=original.map(r=>r.slice());
for(const change of plan.changes){
  const i=ids.indexOf(change.id);
  if(i<0||locked[i]||values[i+1][1]!==change.before)throw Error('Invalid change');
  values[i+1][1]=change.after;
}
sheet.getRange('A1:B6715').values=values; wb.recalculate();
const final=sheet.getRange('A1:B6715').values;
const changed=[];
for(let i=0;i<ids.length;i++){
  const [id,a]=final[i+1];
  if(id!==ids[i]||!['a','b','c','d'].includes(a))throw Error('Invalid output');
  if(locked[i]&&a!==models[0][i])throw Error('Changed consensus');
  if(a!==original[i+1][1])changed.push(id);
}
if(changed.length!==12||JSON.stringify(changed)!==JSON.stringify(plan.changes.map(r=>r.id)))throw Error('Unexpected changes');
if(final[ids.indexOf('test_0047.jpg')+1][1]!=='d')throw Error('0047 changed');
const output=path.join(here,plan.output_file);
const csv=final.map(r=>r.join(',')).join('\n')+'\n';
await fs.writeFile(output,csv,{encoding:'utf8',flag:'wx'});
const reopened=await Workbook.fromCSV(await fs.readFile(output,'utf8'),{sheetName:'Readback'});
if(JSON.stringify(reopened.worksheets.getItemAt(0).getRange('A1:B6715').values)!==JSON.stringify(final))throw Error('Readback differs');
for(const rec of [plan.baseline,...plan.consensus_models])if(sha(await fs.readFile(rec.path))!==rec.sha256)throw Error('Source changed');
const verification={output,sha256:sha(csv),rows:6714,columns:['id','answer'],unique_ids:6714,id_order_preserved:true,valid_answers:true,consensus_rows_preserved:4444,disagreement_rows:2270,changed_rows:12,unchanged_rows:6702,changed_ids:changed,all_changes_in_disagreement:true,source_hashes_unchanged:true,readback_identical:true,submitted:false};
await fs.writeFile(path.join(here,'delivery_verification.json'),JSON.stringify(verification,null,2));
console.log(JSON.stringify(verification));
