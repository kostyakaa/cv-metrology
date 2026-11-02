const fs=require('fs'),path=require('path');
const pptxgen=require('./build_deps/node_modules/pptxgenjs');
const {applyTheme}=require('C:/Users/USER/.agents/skills/synced/10927734-abfd-4dd6-9d70-019bda5e2e15_78e91995-071c-4720-95d0-c6c74ece5c22/pptx/scripts/apply_theme.js');
const slides=JSON.parse(fs.readFileSync(path.join(__dirname,'slides.json'),'utf8'));
const pres=new pptxgen();pres.layout='LAYOUT_WIDE';pres.author='Project review';pres.subject='Аудит камеры и XY-станка';pres.title='Камера и станок — промежуточная презентация';pres.lang='ru-RU';
const THEME={name:'Optical metrology',headFontFace:'Aptos Display',bodyFontFace:'Aptos',colors:{dk1:'122B35',lt1:'F4F6F2',dk2:'455E68',lt2:'E5ECE8',accent1:'087E83',accent2:'DA793B',accent3:'456580',accent4:'85AA9B',accent5:'E2B868',accent6:'72868F',hlink:'087E83',folHlink:'456580'}};
pres.theme={headFontFace:THEME.headFontFace,bodyFontFace:THEME.bodyFontFace,lang:'ru-RU'};const C=pres.SchemeColor;
const W=13.333,H=7.5;
for(const dark of [false,true])pres.defineSlideMaster({title:dark?'Dark':'Light',background:{color:dark?C.text1:C.background1},objects:[
 {placeholder:{options:{name:'title',type:'title',x:.65,y:.8,w:12.05,h:.9,fontSize:29,bold:true,color:dark?C.background1:C.text1,margin:0}}},
 {placeholder:{options:{name:'lead',type:'body',x:.65,y:1.8,w:11.95,h:.62,fontSize:17,color:dark?C.background2:C.text2,margin:0}}},
 {text:{text:'PROJECT PT  /  ОПТИЧЕСКИЕ ИЗМЕРЕНИЯ  /  02.11.2026',options:{x:.65,y:7.08,w:10,h:.18,fontSize:9,color:dark?C.background2:C.text2,margin:0,isTextBox:true}}}
],slideNumber:{x:12,y:7.03,w:.55,h:.25,fontSize:10,color:dark?C.background2:C.text2}});
function text(s,t,x,y,w,h,size=20,color=C.text1,extra={}){s.addText(t,{x,y,w,h,fontSize:size,color,margin:0,breakLine:false,isTextBox:true,...extra});}
function shape(s,type,x,y,w,h,opts={}){s.addShape(type,{x,y,w,h,...opts});}
function target(s,x,y,r,dark=false){shape(s,pres.ShapeType.ellipse,x-r,y-r,r*2,r*2,{line:{color:C.accent1,width:1.2},fill:{color:dark?C.text1:C.background1,transparency:100}});shape(s,pres.ShapeType.line,x-r-.14,y,r*2+.28,0,{line:{color:C.accent1,width:1}});shape(s,pres.ShapeType.line,x,y-r-.14,0,r*2+.28,{line:{color:C.accent1,width:1}});}
function cards(s,a){a.items.forEach((v,i)=>{let x=.65+i*4.08;shape(s,pres.ShapeType.roundRect,x,2.8,3.83,3.27,{radius:.1,rectRadius:.1,line:{color:C.background2},fill:{color:C.background2}});text(s,`0${i+1}`,x+.25,3.04,1,.5,25,C.accent1,{bold:true});text(s,v,x+.25,3.85,3.32,2.0,19,C.text1);});}
const chartOptions={x:.75,y:2.7,w:7.25,h:3.7,showTitle:true,showLegend:false,showValue:true,showCatName:false,catAxisLabelFontFace:'+mn-lt',valAxisLabelFontFace:'+mn-lt',titleFontFace:'+mj-lt',dataLabelFormatCode:'0.000',dataLabelPosition:'outEnd',showBorder:false,showCatName:false,catAxisLabelFontSize:12,valAxisLabelFontSize:11,catAxisLabelColor:C.text2,valAxisLabelColor:C.text2,valGridLine:{color:'E5ECE8',width:1},catGridLine:{style:'none'},chartColors:[THEME.colors.accent1],invertedColors:[THEME.colors.accent2],showCatName:false};
let prev='';
slides.forEach((a,i)=>{
 if(a.section!==prev){pres.addSection({title:a.section});prev=a.section;}
 const dark=a.kind==='cover';const s=pres.addSlide({masterName:dark?'Dark':'Light',sectionTitle:a.section});
 s.addText(a.title,{placeholder:'title'});s.addText(a.lead,{placeholder:'lead'});text(s,a.section.toUpperCase(),.65,.35,11,.22,10,dark?C.accent4:C.accent1,{bold:true,charSpacing:1.6});
 if(a.kind==='cover'){
   a.items.forEach((v,k)=>text(s,v,.75,3.0+k*.9,9.3,.82,22,C.background1));target(s,11.7,4.7,.77,true);target(s,11.7,4.7,.36,true);
 }else if(a.kind==='image'){
   const img=path.join(__dirname,a.image);s.addImage({path:img,x:1,y:2.5,w:a.image.endsWith('jpg')?5.89:6.48,h:4.05,altText:a.title,objectName:'evidence_image'});
   a.items.forEach((v,k)=>text(s,v,9.1,2.8+k*1.12,3.52,.98,17));
 }else if(['calibration','scale','depth'].includes(a.kind)){
   let title,labels,values;
   if(a.kind==='calibration'){title='RMS репроекции, px (40 кадров)';labels=['Pinhole','Rational'];values=[.669285,.548473];}
   if(a.kind==='scale'){title='Модельный недоход на 100 мм, мм';labels=['X','Y'];values=[.634857,.525317];}
   if(a.kind==='depth'){title='Только шум диспаратности: uZ, мм';labels=['База 6 мм','База 12 мм','База 24 мм'];values=[.090367,.045183,.022592];}
   s.addChart(pres.ChartType.bar,[{name:title,labels,values}],{...chartOptions,barDir:'col',title,titleFontSize:15});
   a.items.forEach((v,k)=>text(s,v,8.5,2.85+k*1.12,4.1,1,18));
 }else if(a.kind==='formula'){
   a.items.forEach((v,k)=>{shape(s,pres.ShapeType.roundRect,.7,2.65+k*1.1,11.9,.9,{fill:{color:k===0?C.text1:C.background2},line:{color:k===0?C.text1:C.background2},rectRadius:.08});text(s,v,1,2.88+k*1.1,11.25,.52,22,k===0?C.background1:C.text1);});
 }else if(a.kind==='flow'){
   a.items.forEach((v,k)=>{let x=.7+k*4.1;target(s,x+.3,3.1,.23);text(s,`ЭТАП ${k+1}`,x+.75,2.9,2.7,.5,18,C.accent1,{bold:true});text(s,v,x,3.85,3.65,1.9,23);if(k<2)text(s,'→',x+3.56,3.02,.5,.5,27,C.accent2);});
 }else cards(s,a);
 text(s,a.source,.7,6.65,11.9,.29,10,dark?C.background2:C.text2);
 s.addNotes(a.notes+'\n\nЛокальные источники: '+a.source);
});
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const linkify=s=>esc(s).replace(/https:\/\/[^\s<]+/g,u=>`<a href="${u}">${u}</a>`);
function html(){
const body=slides.map((s,i)=>`<article id="s${i+1}" class="${s.kind==='cover'?'dark':''}"><div class="eyebrow">${esc(s.section)} · ${String(i+1).padStart(2,'0')} / ${slides.length}</div><h1>${esc(s.title)}</h1><p class="lead">${esc(s.lead)}</p>${s.image?`<img class="evidence" src="${s.image}" alt="${esc(s.title)}">`:''}<div class="cards">${s.items.map((v,k)=>`<div><span>0${k+1}</span><p>${esc(v)}</p></div>`).join('')}</div><p class="source">${esc(s.source)}</p><details><summary>Подробные пояснения к слайду</summary><div class="notes">${linkify(s.notes).replaceAll('\n','<br>')}</div></details></article>`).join('\n');
return `<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Камера и станок — анализ проекта</title><style>
*{box-sizing:border-box}body{margin:0;background:#e5ece8;color:#122b35;font:17px/1.65 'Segoe UI',sans-serif}nav{position:sticky;top:0;z-index:2;background:#122b35;color:#f4f6f2;padding:12px 4vw;display:flex;gap:14px;align-items:center;flex-wrap:wrap}nav a{color:#bce4d8;text-decoration:none}button,select{font:inherit;border:1px solid #85aa9b;background:#f4f6f2;color:#122b35;border-radius:7px;padding:5px 12px;cursor:pointer}main{max-width:1180px;margin:auto;padding:28px 20px}article,section{background:#f4f6f2;margin-bottom:28px;padding:45px 50px;border-radius:16px;scroll-margin-top:90px}article.dark{background:#122b35;color:#f4f6f2}.eyebrow{color:#087e83;font-size:12px;text-transform:uppercase;letter-spacing:2px;font-weight:700}.dark .eyebrow{color:#85aa9b}h1{font-size:clamp(26px,3vw,39px);line-height:1.2;margin:16px 0}h2{font-size:28px}.lead{font-size:21px;color:#455e68}.dark .lead{color:#c5d8cf}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin:28px 0}.cards>div{background:#e5ece8;border-radius:10px;padding:22px;color:#122b35}.cards span{font-weight:700;color:#087e83}.cards p{margin-bottom:0}.source{font-size:12px;overflow-wrap:anywhere;color:#455e68}.dark .source{color:#c5d8cf}details{margin-top:22px;border-top:1px solid #85aa9b;padding-top:15px}summary{cursor:pointer;font-weight:600;color:#087e83}.dark summary{color:#bce4d8}.notes{padding-top:18px;max-width:960px;line-height:1.8}.notes a{overflow-wrap:anywhere;color:#087e83}.dark .notes a{color:#bce4d8}.evidence{max-width:100%;max-height:600px;display:block;margin:24px auto;border-radius:10px}.inputs{display:grid;grid-template-columns:repeat(3,1fr);gap:15px}label{font-size:14px}input{display:block;width:100%;padding:8px;font:inherit}output{display:block;background:#122b35;color:white;padding:22px;border-radius:8px;margin-top:20px;font-size:19px}.notice{font-size:14px;color:#455e68}@media(max-width:750px){article,section{padding:28px 22px}.cards,.inputs{grid-template-columns:1fr}nav{position:relative}}@media print{nav,section#calculator{display:none}article{break-before:page;margin:0;border-radius:0;padding:20px}.cards{gap:10px}body{background:white}details{display:block}}
</style><nav><strong>Камера + станок</strong><select id="jump" aria-label="Перейти к слайду">${slides.map((s,i)=>`<option value="s${i+1}">${i+1}. ${esc(s.title)}</option>`).join('')}</select><button id="notes">Раскрыть все пояснения</button><a href="#calculator">Калькулятор точности</a><a href="project_review.pptx">PowerPoint</a></nav><main>${body}
<section id="calculator"><div class="eyebrow">Расчёт чувствительности · не паспорт точности</div><h2>Что изменится при другом шуме, масштабе или базе</h2><div class="inputs">${[['z','Глубина Z, мм',77.238],['f','Фокусное f, px',1100.274],['b','Эффективная база B, мм',12],['sp','Шум одной точки σp, px',.1],['sd','Шум диспаратности σd, px',.1],['l','Длина L, мм',100],['ss','Неопределённость масштаба, %',.1],['sb','Неопределённость базы, мм',.02],['sm','Неопределённость станка, мм',.05]].map(([id,label,v])=>`<label>${label}<input id="${id}" type="number" value="${v}" min="0" step="any"></label>`).join('')}</div><output id="calc"></output><p class="notice">XY: две независимые точки на фронтальной плоскости, общий масштаб; Z: идеальная стереопара, независимые ошибки базы и диспаратности. Не учтены дисторсия, наклон, фокусное, общие корреляции и опорная плоскость. σp и σd задаются отдельно: при независимых одинаковых пиксельных ошибках σd = √2·σp. Выигрыш — условное отношение стандартных неопределённостей, не измеренный результат проекта.</p></section></main><script>
document.getElementById('jump').onchange=e=>document.getElementById(e.target.value).scrollIntoView({behavior:'smooth'});
document.getElementById('notes').onclick=()=>{const ds=[...document.querySelectorAll('details')],open=ds.some(d=>!d.open);ds.forEach(d=>d.open=open);document.getElementById('notes').textContent=open?'Свернуть пояснения':'Раскрыть все пояснения'};
function calculate(){let v=Object.fromEntries([...document.querySelectorAll('input')].map(e=>[e.id,Number(e.value)]));let el=document.getElementById('calc');if(Object.values(v).some(x=>!Number.isFinite(x)||x<0)||v.z<=0||v.f<=0||v.b<=0){el.textContent='Введите неотрицательные значения; Z, f и B должны быть больше нуля.';return}let uxy=Math.hypot(Math.SQRT2*v.z/v.f*v.sp,v.l*v.ss/100),uznoise=v.z*v.z/(v.f*v.b)*v.sd,uzbase=v.z/v.b*v.sb,uz=Math.hypot(uznoise,uzbase);el.textContent='XY: u(L) ≈ '+uxy.toFixed(4)+' мм; относительная u/L: '+(v.l>0?(100*uxy/v.l).toFixed(3)+'%':'не определена')+'. Z: u ≈ '+uz.toFixed(4)+' мм (диспаратность '+uznoise.toFixed(4)+', база '+uzbase.toFixed(4)+'). Условный выигрыш XY: '+(uxy>0?(v.sm/uxy).toFixed(2)+'×':'не определён')+'.'}document.querySelectorAll('input').forEach(e=>e.oninput=calculate);calculate();</script></html>`;
}
(async()=>{await pres.writeFile({fileName:path.join(__dirname,'project_review.pptx')});await applyTheme(path.join(__dirname,'project_review.pptx'),THEME);fs.writeFileSync(path.join(__dirname,'project_review.html'),html(),'utf8');console.log(`Built ${slides.length} slides and HTML`);})();



