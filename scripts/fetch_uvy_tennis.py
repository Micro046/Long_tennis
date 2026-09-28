"""Fetch only tennis videos/labels from the pinned UVY archive using HTTP ranges. Run from project root."""
import io,json,urllib.request,zipfile
from pathlib import Path
URL='https://zenodo.org/api/records/21303900/files/UVY.zip/content'
SIZE=3274165269
class Remote(io.RawIOBase):
 def __init__(self):self.pos=0
 def seekable(self):return True
 def readable(self):return True
 def tell(self):return self.pos
 def seek(self,offset,whence=0):
  self.pos=offset if whence==0 else self.pos+offset if whence==1 else SIZE+offset
  return self.pos
 def read(self,n=-1):
  n=min(n if n>=0 else SIZE-self.pos,SIZE-self.pos)
  if not n:return b''
  if n>12000000:raise ValueError('Refusing unexpected large read')
  start=self.pos;end=start+n-1
  req=urllib.request.Request(URL+f'?range_start={start}',headers={'Range':f'bytes={start}-{end}'})
  with urllib.request.urlopen(req,timeout=30) as r:
   if r.status!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {start}-'):raise ValueError('Range not respected')
   b=r.read(n)
  self.pos+=len(b);return b

def main():
 import hashlib
 out=Path('data/public/uvy');out.mkdir(parents=True,exist_ok=True)
 record=json.load(urllib.request.urlopen('https://zenodo.org/api/records/21303900',timeout=20))
 (out/'source_record.json').write_text(json.dumps(record,indent=2))
 manifest={'source':'https://zenodo.org/records/21303900','doi':'10.5281/zenodo.21303900','licenseDeclaredByPublisher':'CC-BY-4.0','creators':['Elton Alencar','Rosiane de Freitas'],'purpose':'External identity-association evaluation; spectator footage, phone device unverified','files':[]}
 with zipfile.ZipFile(Remote()) as z:
  for i in z.infolist():
   parts=Path(i.filename).parts
   if len(parts)<3 or parts[1] not in ('tennis_V01','tennis_V02','tennis_V03'):continue
   if not (i.filename.endswith('.mp4') or Path(i.filename).name in ('video_info.txt','gt.txt','labels.txt')):continue
   dest=out/parts[1]/Path(i.filename).name;dest.parent.mkdir(parents=True,exist_ok=True)
   content=z.read(i);dest.write_bytes(content)
   manifest['files'].append({'archiveMember':i.filename,'path':str(dest),'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content)})
   print(dest,len(content),flush=True)
 (out/'PROVENANCE.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
