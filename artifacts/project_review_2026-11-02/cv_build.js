// Design workflow adapted from unslop by Damir Sokolovsky, CC BY 4.0.
// https://github.com/heydamir/unslop
const fs=require('fs'),path=require('path');
const pptxgen=require('./build_deps/node_modules/pptxgenjs');
const {applyTheme}=require('C:/Users/USER/.agents/skills/synced/10927734-abfd-4dd6-9d70-019bda5e2e15_78e91995-071c-4720-95d0-c6c74ece5c22/pptx/scripts/apply_theme.js');
const p=new pptxgen();p.layout='LAYOUT_WIDE';p.title='Камера и станок — промежуточные результаты';p.author='Проект PT';p.lang='ru-RU';
const theme={name:'Измерительная установка',headFontFace:'Aptos',bodyFontFace:'Aptos',colors:{dk1:'202426',dk2:'535B60',lt1:'FAFAF8',lt2:'E9ECEB',accent1:'176B69',accent2:'A06037',accent3:'607078',accent4:'8AA69C',accent5:'BCB4A7',accent6:'78858A',hlink:'176B69',folHlink:'535B60'}};
p.theme={headFontFace:theme.headFontFace,bodyFontFace:theme.bodyFontFace};const C=p.SchemeColor;
p.defineSlideMaster({title:'Report',background:{color:C.background1},objects:[{placeholder:{options:{name:'title',type:'title',x:.67,y:.67,w:12,h:1.15,fontSize:32,bold:false,align:'left',margin:0,color:C.text1}}},{text:{text:'Камера и станок · промежуточные результаты · ноябрь 2026',options:{x:.67,y:7.02,w:10,h:.22,fontSize:10,color:C.text2,margin:0,isTextBox:true}}}],slideNumber:{x:12,y:6.97,w:.5,h:.3,color:C.text2,fontSize:11}});
const data=JSON.parse(fs.readFileSync(path.join(__dirname,'cv_content.json'),'utf8'));
function txt(s,t,x,y,w,h,size=22,c=C.text1){s.addText(t,{x,y,w,h,fontSize:size,color:c,margin:0,isTextBox:true,breakLine:false});}
function line(s,x,y,w){s.addShape(p.ShapeType.line,{x,y,w,h:0,line:{color:C.background2,width:1}});}
function photo(s,file,x,y,w,h){s.addImage({path:file,x,y,w,h,altText:'Сохранённый результат испытаний проекта'});}
const root=path.resolve(__dirname,'../..');
p.addSection({title:'Промежуточные результаты'});
data.forEach(([title,lead,items,notes,kind,img],i)=>{
const s=p.addSlide({masterName:'Report',sectionTitle:'Промежуточные результаты'});s.addText(title,{placeholder:'title'});txt(s,lead,.67,1.85,11.9,.85,20,C.text2);
if(kind==='table'){
items.forEach((v,j)=>{const parts=v.split(' | ');txt(s,parts[0],.67,2.95+j*.87,3.2,.65,19,C.accent1);txt(s,parts[1]||'',4,2.95+j*.87,8.6,.72,20);if(j<3)line(s,.67,3.72+j*.87,12);});
}else if(kind==='evidence'){
photo(s,path.join(root,img),.67,3.0,6.3,3.544);items.forEach((v,j)=>txt(s,v,7.45,2.85+j*.94,5.2,.88,18));
}else if(kind==='cover'){
photo(s,path.join(root,'scans/scan_20261019-190156/heights.png'),6.3,3.05,6.3,3.54);items.forEach((v,j)=>txt(s,v,.67,3.1+j*.9,5,.85,22));
}else if(kind==='process'){
items.forEach((v,j)=>{let x=.67+j*4.22;txt(s,String(j+1),x,3.05,.65,.65,36,C.accent1);txt(s,v,x,4.05,3.55,1.85,24);if(j<2)txt(s,'→',x+3.55,3.2,.6,.5,24,C.text2);});
}else if(kind==='calibration'){
photo(s,path.join(root,'calib/frames_cam0_1920x1080_20261019-153432/view_30.png'),.67,3,6.5,3.656);items.forEach((v,j)=>txt(s,v,7.7,3.1+j*1.05,4.7,.9,22));
}else if(kind==='numbers'){
txt(s,'100 мм',.67,3.0,4,.8,44);txt(s,'задано станку',.67,3.95,4,.6,20,C.text2);
txt(s,'99,37 мм',5,3,3.5,.8,40,C.accent1);txt(s,'по X',5,3.95,3.5,.5,20,C.text2);txt(s,'99,47 мм',9,3,3.7,.8,40,C.accent1);txt(s,'по Y',9,3.95,3.5,.5,20,C.text2);
line(s,.67,5.05,12);txt(s,items[2],.67,5.45,11.8,.9,23);
}else if(kind==='closure'){
txt(s,'0,027 мм',.67,3,5,1.1,60,C.accent1);txt(s,'разница при возврате\nк исходной метке',.67,4.3,5,1,24,C.text2);items.forEach((v,j)=>txt(s,v,6.4,3.1+j*1.1,5.8,.9,22));
}else if(kind==='height'){
photo(s,path.join(root,'scans/scan_20261019-185902/heights.png'),.67,3,7,3.9375);items.forEach((v,j)=>txt(s,v,8.2,3.1+j*1.05,4.4,1.0,20));
}else if(kind==='limits'){
txt(s,'30–64%',.67,3,5,1.1,60,C.accent1);txt(s,'доля кадра, где удалось\nопределить высоту',.67,4.3,5,1,24,C.text2);items.forEach((v,j)=>txt(s,v,6.4,3+j*1.15,5.8,1.05,20));
}else{
items.forEach((v,j)=>{txt(s,String(j+1),.67,3+j*1.12,.5,.5,22,C.accent1);txt(s,v,1.5,3+j*1.12,10.9,.85,25);if(j<2)line(s,1.5,3.95+j*1.12,10.9);});
}
s.addNotes(notes+'\n\nОснование: исходники проекта; calib/table_frame.json; calib/calib_cam0_1920x1080_pinhole_rational.json; scans/; log.txt.\nОформление переработано по unslop by Damir Sokolovsky, CC BY 4.0: https://github.com/heydamir/unslop');
});
let md='# Камера и станок: доклад о промежуточных результатах\n\nВерсия для заказчика. 9 слайдов, ориентировочно 8–10 минут.\n\n';
data.forEach(([t,l,items,n],i)=>{md+=`## Слайд ${i+1}. ${t}\n\n${n}\n\n`;});
md+='## Если спросят о технологии\n\nСистема написана на Python. OpenCV обрабатывает изображения и распознаёт печатные метки, NumPy выполняет расчёты. Обмен со станком идёт через последовательный порт по протоколу GRBL 1.1. Используется обычная камера, перемещаемая по двум осям. Нейросетевая модель для измерений не требуется.\n\n## Как отвечать на вопрос о точности\n\nТочность ещё проверяется. 0,027 мм — разница при возврате в одном обходе, а не гарантированная ошибка любых измерений. Расхождение масштаба 0,5–0,6% найдено относительно печатной доски, размер которой нужно независимо проверить. Сколько даёт совместная коррекция камеры и станка, покажет следующая серия испытаний.\n\n## Основания и оформление\n\nЧисла взяты из сохранённых калибровок, сканов и журнала проекта; новых аппаратных испытаний при подготовке доклада не проводилось. Полный технический разбор сохранён отдельно в project_description.md.\n\nИспользован и адаптирован к презентации навык [unslop](https://github.com/heydamir/unslop) — Damir Sokolovsky, CC BY 4.0. Убраны повторяющиеся карточки, декоративные прицелы, заголовки прописными буквами и необъяснённые формулы. Цвет оставлен в реальных результатах измерения и ключевых числах. Изображения не генерировались.\n';
fs.writeFileSync(path.join(__dirname,'cv_talk.md'),md,'utf8');
(async()=>{await p.writeFile({fileName:path.join(__dirname,'cv_results.pptx')});await applyTheme(path.join(__dirname,'cv_results.pptx'),theme);console.log('Customer deck and talk saved');})();


